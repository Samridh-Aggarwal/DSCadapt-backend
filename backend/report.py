"""Turn the JSONL traces into something a person can read.

The old logger rebuilt an HTML report over every query ever recorded, inside
each request, before returning an answer. That is why it could not be asked to
capture more: the cost grew with the number of answers already given.

Writing and reading are separated now. logger.py appends one line per query,
which takes a tenth of a millisecond. This renders the report only when
somebody opens it.

Same information as before and rather more of it — the settings that were
never captured, retrieval scores split into their halves, whether a refusal
happened, and every output-rule violation.
"""

import html as html_
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROUTE_COLOURS = {"domain": "#0f6e56", "both": "#534ab7", "web": "#854f0b",
                 "meta": "#185fa5", "redirect": "#993536", "empty": "#5f5e5a"}


def read(directory, limit=500):
    """Newest traces first, across every day and every process."""
    records = []
    for path in sorted(Path(directory).rglob("*.jsonl"), reverse=True):
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                return records
    return records


def esc(value):
    return html_.escape(str(value if value is not None else ""))


def render(records):
    if not records:
        return _page("<p class='empty'>No traces yet. Ask the tool something and refresh.</p>", {})

    sessions = defaultdict(list)
    for record in records:
        sessions[record.get("session") or record.get("boot", "—")].append(record)

    asks = [r for r in records if r.get("kind") == "ask"]
    totals = {
        "queries": len(asks),
        "sessions": len(sessions),
        "cost": sum(r.get("cost", {}).get("total", 0) for r in asks),
        "redirects": sum(1 for r in asks if r.get("redirected")),
        "violations": sum(len([v for v in r.get("violations", []) if not v["advisory"]])
                          for r in asks),
        "surveillance": sum(1 for r in asks if r.get("surveillance", {}).get("used")),
    }

    body = []
    for name, entries in sessions.items():
        stamp = (entries[0].get("ts") or "")[:16].replace("T", " ")
        body.append(f"<div class='session'><div class='session-head'>"
                    f"<b>{esc(name)}</b><span>{len(entries)} queries · {esc(stamp)}</span></div>")
        for record in entries:
            body.append(_entry(record))
        body.append("</div>")
    return _page("".join(body), totals)


def _entry(record):
    if record.get("kind") == "error":
        return (f"<details class='q err'><summary><span class='tag' "
                f"style='background:#993536'>error</span>"
                f"<span class='qt'>{esc(record.get('stage'))}: "
                f"{esc(record.get('error', {}).get('message'))}</span></summary></details>")
    if record.get("kind") in ("review", "briefing"):
        return (f"<details class='q'><summary><span class='tag' style='background:#5f5e5a'>"
                f"{esc(record['kind'])}</span><span class='qt'>"
                f"{esc(json.dumps({k: v for k, v in record.items() if k not in ('kind','ts','boot','version','draft','final','comments')})[:160])}"
                f"</span></summary><pre>{esc(json.dumps(record, indent=2)[:6000])}</pre></details>")

    router = record.get("router", {})
    route = router.get("route", "?")
    retrieval = record.get("retrieval", {})
    settings = record.get("settings", {})
    timing = record.get("timing", {})
    hard = [v for v in record.get("violations", []) if not v["advisory"]]
    soft = [v for v in record.get("violations", []) if v["advisory"]]

    parts = [f"<details class='q'><summary>"
             f"<span class='tag' style='background:{ROUTE_COLOURS.get(route, '#5f5e5a')}'>"
             f"{esc(route)}</span>"
             f"<span class='qt'>{esc(record.get('message') or '(text not captured)')}</span>"
             f"<span class='meta'>{timing.get('total', '?')}s · "
             f"${record.get('cost', {}).get('total', 0):.4f}</span></summary><div class='detail'>"]

    chips = [f"<i>{esc(k)}</i> {esc(v)}" for k, v in settings.items()]
    if not router.get("parsed", True):
        chips.append("<b style='color:#e24b4a'>router did not parse</b>")
    if record.get("prompt_sha"):
        chips.append(f"<i>prompt</i> {esc(record['prompt_sha'])}")
    parts.append("<div class='chips'>" + " ".join(f"<span>{c}</span>" for c in chips) + "</div>")

    if router.get("rewritten") and router.get("rewrote"):
        parts.append(f"<div class='sec'><h4>rewritten</h4><p>{esc(router['rewritten'])}</p></div>")

    if retrieval.get("hits"):
        rows = "".join(
            f"<tr><td class='n'>{h['score']:.3f}</td><td class='n dim'>{h['cosine']:.3f}</td>"
            f"<td class='n dim'>{h['keyword']:.3f}</td><td>{esc(h['section'])}"
            f"<div class='dim'>{esc(h['source'])}</div></td></tr>"
            for h in retrieval["hits"])
        thin = " · <b style='color:#ba7517'>thin coverage</b>" if retrieval.get("thin_coverage") else ""
        parts.append(f"<div class='sec'><h4>retrieval — {retrieval['n']} chunks{thin}</h4>"
                     f"<table><tr><th>score</th><th>cosine</th><th>keyword</th><th>section</th></tr>"
                     f"{rows}</table></div>")

    surveillance = record.get("surveillance", {})
    if surveillance.get("diseases"):
        used = ("<b style='color:#0f6e56'>injected</b>" if surveillance.get("used")
                else "<b style='color:#ba7517'>detected but no data for this country</b>")
        parts.append(f"<div class='sec'><h4>surveillance — {used}</h4>"
                     f"<p>{esc(', '.join(surveillance['diseases']))}</p></div>")

    web = record.get("web", {})
    if web.get("attempted"):
        state = f"{web['chars']} chars" if web.get("succeeded") else "<b style='color:#e24b4a'>returned nothing</b>"
        parts.append(f"<div class='sec'><h4>web search</h4><p>{state}</p></div>")

    for key, label in [("publications", "publications"), ("scoping", "scoping review")]:
        if record.get(key):
            parts.append(f"<div class='sec'><h4>{label}</h4>"
                         f"<p>{esc(' · '.join(record[key]))}</p></div>")

    if record.get("answer"):
        parts.append(f"<div class='sec'><h4>answer — "
                     f"{record.get('generation', {}).get('words', 0)} words, "
                     f"{record.get('generation', {}).get('context_chars', 0)} chars of context"
                     f"</h4><div class='ans'>{esc(record['answer'])}</div></div>")

    if hard or soft:
        rows = "".join("<div class='v adv'>" if v["advisory"] else "<div class='v'>"
                       + f"<b>{esc(v['rule'])}</b> {esc(v['detail'])}</div>"
                       for v in hard + soft)
        parts.append(f"<div class='sec'><h4>violations — {len(hard)} hard, "
                     f"{len(soft)} advisory</h4>{rows}</div>")

    attribution = record.get("attribution")
    if attribution:
        counts = " · ".join(f"{k} {v}" for k, v in attribution.get("counts", {}).items()) or "none"
        parts.append(f"<div class='sec'><h4>highlighting</h4><p>{esc(counts)} "
                     f"— {attribution.get('scored', 0)} sentences scored, "
                     f"{attribution.get('skipped_short', 0)} too short, "
                     f"{attribution.get('embedded', 0)} embeddings</p></div>")

    stages = " ".join(f"<span><i>{esc(k)}</i> {esc(v)}s</span>" for k, v in timing.items())
    parts.append(f"<div class='chips'>{stages}</div></div></details>")
    return "".join(parts)


def _page(body, totals):
    stats = "".join(
        f"<div class='stat'><span>{k}</span><b>{v:.4f}</b></div>" if k == "cost"
        else f"<div class='stat'><span>{k}</span><b>{v}</b></div>"
        for k, v in totals.items())
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>DSCAdapt traces</title><style>
*{{box-sizing:border-box}}
body{{background:#16161a;color:#dedcd4;font:14px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px}}
h1{{font-size:19px;font-weight:500;color:#fff;margin:0 0 4px}}
.sub{{color:#8d8b84;font-size:13px;margin:0 0 20px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px}}
.stat{{background:#1e1e24;border-radius:8px;padding:10px 14px;min-width:96px}}
.stat span{{display:block;color:#8d8b84;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
.stat b{{font-size:20px;font-weight:500;color:#fff}}
.session{{margin-bottom:16px;border:1px solid #2e2e35;border-radius:10px;overflow:hidden}}
.session-head{{background:#1e1e24;padding:9px 14px;display:flex;justify-content:space-between;font-size:13px}}
.session-head span{{color:#8d8b84}}
.q{{border-top:1px solid #2e2e35}}
.q summary{{padding:9px 14px;cursor:pointer;display:flex;gap:9px;align-items:center;list-style:none}}
.q summary::-webkit-details-marker{{display:none}}
.q[open] summary{{background:#1a1a20}}
.tag{{font-size:10px;font-weight:500;padding:2px 8px;border-radius:10px;color:#fff;flex-shrink:0}}
.qt{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.meta{{color:#8d8b84;font-size:12px;flex-shrink:0}}
.detail{{padding:4px 14px 14px}}
.sec{{margin-top:12px}}
.sec h4{{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#8d8b84;font-weight:500;margin:0 0 6px}}
.sec p{{margin:0}}
.ans{{background:#1e1e24;border-radius:6px;padding:11px 13px;white-space:pre-wrap;font-size:13px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;color:#8d8b84;font-weight:500;font-size:11px;padding:3px 8px 5px}}
td{{padding:4px 8px;border-top:1px solid #24242a;vertical-align:top}}
td.n{{font-variant-numeric:tabular-nums;width:58px;color:#5dcaa5}}
.dim{{color:#8d8b84;font-size:11.5px}}
.chips{{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}}
.chips span{{background:#1e1e24;border-radius:4px;padding:3px 8px;font-size:11.5px}}
.chips i{{color:#8d8b84;font-style:normal}}
.v{{background:#2a1a1a;border-left:2px solid #e24b4a;padding:5px 9px;margin-bottom:3px;font-size:12.5px}}
.v.adv{{background:#262018;border-left-color:#ba7517}}
pre{{background:#1e1e24;padding:11px;border-radius:6px;overflow-x:auto;font-size:12px}}
.empty{{color:#8d8b84}}
</style></head><body>
<h1>DSCAdapt traces</h1>
<p class="sub">{datetime.now():%d %B %Y, %H:%M} · newest first · rendered on request, not on every query</p>
<div class="stats">{stats}</div>
{body}</body></html>"""
