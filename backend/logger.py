"""Per-query traces, written locally and pushed in the background.

The old logger did two Hugging Face uploads and regenerated an HTML report
covering every query ever recorded, synchronously, inside the request. Every
answer paid for it, and the cost grew with the number of answers already given.
Which meant it could not be asked to capture more, and it was missing most of
what an evaluation needs.

So transport and capture are both replaced.

Transport: one line appended to a local JSONL per query, which is microseconds,
and a background thread that uploads the file every so often. Each process
writes its own file — logs/<date>/<boot>.jsonl — so there is no read, modify,
write cycle and two instances cannot overwrite each other.

Capture: the settings that were never recorded at all, so traffic could not be
sliced by country or language; the retrieval scores split into their cosine and
keyword halves, so the 0.85/0.15 blend can be tuned from real queries; whether
a refusal happened and where it came from; a hash of the prompt, so a change in
the numbers can be traced to the change in the code that caused it; and the
timing of each stage rather than only the total.

The point is not observability for its own sake. A trace that carries the
question, the retrieved sections, the answer and the settings is a golden-set
entry that someone already wrote for you.
"""

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

COST_PER_MILLION = {
    "ministral-8b-latest": {"input": 0.15, "output": 0.15},
    "mistral-small-latest": {"input": 0.15, "output": 0.60},
    "mistral-medium-latest": {"input": 1.50, "output": 7.50},
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-embed": {"input": 0.10, "output": 0.00},
}


def code_version():
    """Something that changes when the code changes.

    Without it a shift in the numbers cannot be attributed to anything. Git
    first; Spaces have no repository, so an environment variable next.
    """
    env = os.environ.get("CODE_VERSION")
    if env:
        return env
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=2,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def prompt_fingerprint(text):
    """Twelve characters of the prompt. Editing it moves the number."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def cost_of(model, usage):
    if not usage:
        return 0.0
    rates = COST_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    prompt = usage.get("input", 0) or 0
    completion = usage.get("output", 0) or 0
    return round((prompt * rates["input"] + completion * rates["output"]) / 1_000_000, 6)


# --- where traces go ---------------------------------------------------------

class NullSink:
    """No upload. The default, so nothing is published by accident."""

    def send(self, path):
        pass


class HuggingFaceSink:
    """Upload the current file to a dataset.

    The repository is a parameter. It used to be a constant pointing at one
    person's private dataset, so anyone forking this and running it with their
    own token would have written their traffic into someone else's repo.
    """

    def __init__(self, repo_id, token, path_prefix="logs"):
        self.repo_id = repo_id
        self.token = token
        self.prefix = path_prefix

    def send(self, path):
        from huggingface_hub import HfApi
        HfApi(token=self.token).upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{self.prefix}/{path.parent.name}/{path.name}",
            repo_id=self.repo_id, repo_type="dataset",
            commit_message=f"traces {path.parent.name}")


class Logger:
    def __init__(self, directory="/tmp/dscadapt-logs", sink=None,
                 batch=20, flush_seconds=120, echo=False):
        self.directory = Path(directory)
        self.sink = sink or NullSink()
        self.batch = batch
        self.flush_seconds = flush_seconds
        self.echo = echo

        self.boot = uuid.uuid4().hex[:8]
        self.version = code_version()
        self.written = 0
        self.failures = 0

        self._pending = 0
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._errors = queue.Queue(maxsize=32)

        self.directory.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(target=self._run, name="logger", daemon=True)
        self._worker.start()

    @property
    def path(self):
        day = self.directory / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        return day / f"{self.boot}.jsonl"

    @property
    def readable_path(self):
        """The same records, indented, for reading in a browser.

        One line per query is right for appending and wrong for reading — the
        whole trace arrives as a single enormous line. This is written from the
        JSONL at flush time, on the background thread, so a person gets a file
        they can actually open without any of it happening inside a request.

        One file per process rather than one growing file. The original was a
        single all_logs.json that reached 3.25 MB, at which point the Hub
        stopped rendering it at all.
        """
        return self.path.with_suffix(".json")

    def _write_readable(self):
        try:
            records = [json.loads(line) for line in
                       self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.readable_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            return self.readable_path
        except Exception as err:
            print(f"[LOGGER] could not write the readable copy: {err}")
            return None

    def record(self, entry):
        """Append one trace. Never raises — a logging problem is not a user's problem."""
        try:
            entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
            entry.setdefault("version", self.version)
            entry.setdefault("boot", self.boot)
            line = json.dumps(entry, default=str, ensure_ascii=False)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                self.written += 1
                self._pending += 1
                due = (self._pending >= self.batch
                       or time.time() - self._last_flush > self.flush_seconds)
            if self.echo:
                print(summarise(entry))
            if due:
                self._wake.set()
        except Exception as err:
            self.failures += 1
            print(f"[LOGGER] could not record: {err}")

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=self.flush_seconds)
            self._wake.clear()
            self._upload()

    def _upload(self):
        with self._lock:
            if self._pending == 0:
                return
            path, self._pending, self._last_flush = self.path, 0, time.time()
        try:
            readable = self._write_readable()
            self.sink.send(path)
            if readable:
                self.sink.send(readable)
        except Exception as err:
            self.failures += 1
            try:
                self._errors.put_nowait(str(err))
            except queue.Full:
                pass
            print(f"[LOGGER] upload failed, the local file is intact: {err}")

    def flush(self):
        """Blocking. Called at shutdown."""
        self._upload()

    def close(self):
        self._stop.set()
        self._wake.set()
        self.flush()

    def health(self):
        return {"written": self.written, "pending": self._pending,
                "failures": self.failures, "boot": self.boot, "version": self.version,
                "sink": type(self.sink).__name__}


# --- what a trace holds ------------------------------------------------------

def query_record(message, answer, audience="Policymaker", country="All Europe",
                 language="English", length="Standard", session="", turn=1,
                 attribution=None, system_prompt="", capture_text=True):
    """One /ask trace.

    capture_text is a switch rather than an assumption. Questions and answers
    are what make these traces an evaluation asset, and they are also the part
    a consent notice has to cover.
    """
    hits = answer.hits or []
    record = {
        "kind": "ask",
        "session": session,
        "turn": turn,
        "prompt_sha": prompt_fingerprint(system_prompt) if system_prompt else None,

        # never captured before, so traffic could not be sliced by any of them
        "settings": {"audience": audience, "country": country,
                     "language": language, "length": length},

        "router": {
            "route": answer.route,
            "in_scope": answer.in_scope,
            "parsed": answer.router_parsed,
            "rewritten": answer.rewritten if capture_text else None,
            "rewrote": answer.rewritten.strip().lower() != message.strip().lower(),
        },
        # A redirect is a route now. It used to be produced by the model, which
        # logged it as a normal domain answer with a passage count, so refusals
        # could not be counted.
        "redirected": answer.route == "redirect",

        "retrieval": {
            "n": len(hits),
            "thin_coverage": answer.thin_coverage,
            "sources": sorted({h.chunk.source for h in hits}),
            # split, so the blend can be tuned from real queries rather than guessed
            "hits": [{"section": h.chunk.section, "source": h.chunk.source,
                      "score": round(h.score, 4), "cosine": round(h.cosine, 4),
                      "keyword": round(h.keyword, 4)} for h in hits],
        },
        "web": {
            "attempted": bool(answer.web_text) or answer.web_failed,
            "succeeded": bool(answer.web_text),
            "chars": len(answer.web_text),
        },
        "surveillance": {
            "diseases": answer.diseases,
            "used": answer.used_surveillance,
            "block_chars": len(answer.ecdc_block),
        },
        "publications": [r["short"] for r, _ in (answer.publications or [])],
        "scoping": [r["section"] for r, _ in (answer.scoping or [])],

        "generation": {
            "words": len(answer.text.split()),
            "context_chars": answer.context_chars,
        },
        "violations": [{"rule": v.rule, "detail": v.detail, "advisory": v.advisory}
                       for v in answer.violations],
        "timing": answer.timing,
        "tokens": answer.usage,
        "cost": {stage: cost_of(_model_for(stage), usage)
                 for stage, usage in (answer.usage or {}).items()},
    }
    record["cost"]["total"] = round(sum(record["cost"].values()), 6)

    if attribution is not None:
        record["attribution"] = {
            "counts": attribution.counts, "scored": attribution.scored,
            "skipped_short": attribution.skipped_short, "embedded": attribution.embedded,
        }
    if capture_text:
        record["message"] = message
        record["answer"] = answer.text
    else:
        record["message_chars"] = len(message)
    return record


def error_record(message, stage, err, session="", turn=1, **settings):
    return {
        "kind": "error", "session": session, "turn": turn,
        "settings": settings,
        "stage": stage,
        "error": {"type": type(err).__name__, "message": str(err)[:500]},
        "message_chars": len(message),
    }


def _model_for(stage):
    return {"router": "ministral-8b-latest", "embedding": "mistral-embed",
            "generation": "mistral-small-latest", "web_search": "mistral-medium-latest",
            "eval_scope": "ministral-8b-latest", "review": "mistral-small-latest",
            }.get(stage, stage)


def summarise(entry):
    """One line to stdout, for watching a Space's logs."""
    if entry.get("kind") == "error":
        return f"  ERROR {entry['stage']}: {entry['error']['message'][:80]}"
    router = entry.get("router", {})
    retrieval = entry.get("retrieval", {})
    hard = [v for v in entry.get("violations", []) if not v["advisory"]]
    bits = [
        f"{router.get('route','?'):<8}",
        f"{retrieval.get('n',0)} chunks",
        f"{entry.get('generation',{}).get('words',0)}w",
        f"{entry.get('timing',{}).get('total','?')}s",
        f"${entry.get('cost',{}).get('total',0):.4f}",
    ]
    if entry.get("surveillance", {}).get("used"):
        bits.append("ecdc")
    if not router.get("parsed", True):
        bits.append("ROUTER UNPARSED")
    if hard:
        bits.append("violations: " + ",".join(sorted({v["rule"] for v in hard})))
    return "  " + " | ".join(bits)
