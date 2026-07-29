"""One question in, one answer out.

Route, gather evidence, assemble the context, generate, check the output.

The change that matters here is what "route" decides. It used to decide which
sources existed at all: the query vector was created inside retrieve(), so a
web-routed question had no vector, and everything keyed off one — surveillance
data, project publications, the scoping review — was skipped along with it. The
reference list too, since that came from the retrieved chunks. A question like
"what is the current West Nile situation in Italy", which is exactly what the
web route is for, got web text and nothing else.

Route now decides only where the prose comes from. Which country the user
picked, which diseases are in play, and which publications are relevant are all
independent of that, so they are gathered every time.

The disease scan reads the web results as well. It only ever saw the query and
the retrieved chunks, and web results are full of disease names — that is what
they are for.

Everything else here is deliberately unchanged. Same models, same temperatures,
same top-k, same context ordering, same scoring. Those all need the eval set.
"""

import re
import time
from dataclasses import dataclass, field

import ecdc
import postprocess
import prompts
import retrieval

GENERATION_MODEL = "mistral-small-latest"
ROUTER_MODEL = "ministral-8b-latest"

TOP_K_DOMAIN = 10        # route=domain: no web results competing for context
TOP_K_BOTH = 7           # route=both: leave room for the web block
REASONING_HEADROOM = 3000
LENGTH_TOKENS = {"Brief": 500, "Standard": 1200, "Detailed": 2400}

# Below this, the corpus does not really cover the question. Set by eye and
# unvalidated — it wants calibrating against the golden set before it is
# trusted, so it sits low enough to fire rarely. Cosine rather than the blended
# score, because the keyword half is relative to this corpus and this query,
# while cosine answers the question actually being asked: is there anything in
# here about this.
THIN_COVERAGE_COSINE = 0.70

TAGS = re.compile(r"<[^>]+>")


class PipelineError(RuntimeError):
    """Something failed that cannot be papered over.

    app.py used to catch generation errors and return the exception text as the
    answer, at HTTP 200. Users saw "Error: 429 rate limit exceeded" in the chat
    bubble, monitoring saw a healthy response, and the logs recorded a
    successful answer of four words.
    """


@dataclass
class Answer:
    text: str = ""
    route: str = "domain"
    rewritten: str = ""
    in_scope: bool = True
    router_parsed: bool = True

    hits: list = field(default_factory=list)
    references: dict = field(default_factory=dict)
    web_text: str = ""
    web_failed: bool = False
    thin_coverage: bool = False

    diseases: list = field(default_factory=list)
    ecdc_block: str = ""
    publications: list = field(default_factory=list)
    scoping: list = field(default_factory=list)

    context_chars: int = 0
    violations: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def used_surveillance(self):
        """Whether surveillance text actually reached the model.

        Not the same as whether diseases were detected. /ask built its ECDC
        source citation from the disease list, which is populated even when the
        block comes back empty — so a UK user saw a citation to the ECDC Atlas
        for data that was never injected, because the UK is not in the extract.
        """
        return bool(self.ecdc_block)


# --- router ------------------------------------------------------------------

def parse_router(raw, fallback):
    """Three lines. Returns the result and whether it actually parsed.

    Failing open on scope is right — wrongly refusing a real question is worse
    than answering a stray one — but it has to be visible. The original
    defaulted silently, so a router that had stopped working looked exactly
    like a router waving everything through.
    """
    result = {"rewritten": fallback, "in_scope": True, "route": "domain"}
    seen = set()
    for line in raw.split("\n"):
        line = line.strip()
        low = line.lower()
        if low.startswith("rewritten:"):
            value = line.split(":", 1)[1].strip()
            if value:
                result["rewritten"] = value
                seen.add("rewritten")
        elif low.startswith("in_scope:"):
            result["in_scope"] = line.split(":", 1)[1].strip().lower() in ("yes", "true")
            seen.add("in_scope")
        elif low.startswith("route:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in ("domain", "web", "both", "meta"):
                result["route"] = value
                seen.add("route")
    return result, seen == {"rewritten", "in_scope", "route"}


def clean_history(history, limit=None):
    """Strip the markup the frontend added before the model sees it.

    Answers are stored with their highlight spans, and those were fed straight
    back in. The model read its own <span class="highlight"> tags while being
    told its output must contain no markup.
    """
    out = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue                      # a null in the array is not worth a 500
        content = entry.get("content")
        if not isinstance(content, str) or entry.get("role") not in ("user", "assistant"):
            continue
        content = TAGS.sub("", content).strip()
        if content:
            out.append({"role": entry["role"], "content": content})
    return out[-limit:] if limit else out


# --- context -----------------------------------------------------------------

def build_context(hits, references, web_text="", ecdc_block="", publications=None,
                  scoping=None, web_failed=False, thin_coverage=False):
    """Assemble what the model reads.

    Section order is unchanged from the original. Where a block sits in the
    context affects how much attention it gets, and moving surveillance data
    up — which is arguably right, since it overrules the other sources on
    whether a disease is present — is a change that needs measuring.
    """
    parts = []

    if hits:
        parts.append("EVIDENCE FROM IDALERT PATHWAY DOCUMENTS:\n")
        for hit in hits:
            parts.append(f"--- {hit.chunk.section} ---")
            parts.append(hit.chunk.text + "\n")
        if references:
            parts.append("\nREFERENCE LIST — cite by author/organisation and year:\n")
            for source, text in references.items():
                short = source.replace(".docx.txt", "").replace(".txt", "")
                parts.append(f"--- References from: {short} ---")
                parts.append(text[:4000] + "\n")

    for record, _ in (scoping or []):
        parts.append("\n[IDAlert WP2 SCOPING REVIEW (D2.2, 2024) — higher-level synthesis of "
                     "climate policy and infectious disease; cite as 'IDAlert D2.2 scoping "
                     "review (2024)' if you draw on it]\n")
        parts.append(f"--- {record['section']} ---")
        parts.append(record["text"] + "\n")

    if ecdc_block:
        parts.append("\n" + ecdc_block)

    if web_text:
        parts.append("\nSUPPLEMENTARY WEB SEARCH RESULTS (current/local context — treat exact "
                     "specifics with caution):\n")
        parts.append("These came from a web search and can be wrong on exact specifics. Use them "
                     "for substance and direction only. Do not reproduce any specific regulation, "
                     "directive or decree number, article number, exact enactment date, or "
                     "precise statistic from them as fact, and do not put such identifiers in "
                     "your reference list. Name the law or policy, attribute it to web sources, "
                     "and note the exact citation should be verified.\n")
        parts.append(web_text + "\n")
    elif web_failed:
        parts.append("\n" + prompts.WEB_FAILED + "\n")

    if thin_coverage:
        parts.append("\n" + prompts.THIN_COVERAGE + "\n")

    for record, _ in (publications or []):
        parts.append("\n[IDAlert PROJECT PUBLICATION - cite in your response and reference list "
                     "if relevant]\n")
        parts.append(f"{record['short']}. {record['title']}.")
        parts.append(f"Link: {record['link']}\n")
        parts.append(record["text"] + "\n")

    if not parts:
        return ""
    return "<retrieved_context>\n\n" + "\n".join(parts) + "\n</retrieved_context>"


def author_key(short_citation):
    """The name to look for in the answer, from a short citation.

    "Parvage et al. (2025)" -> "parvage". The original took the last token
    before the bracket, which is "al." for every multi-author paper, so the
    check was looking for a string that has nothing to do with the author.
    """
    name = short_citation.split("(")[0]
    name = re.sub(r"\s+et\s+al\.?\s*$", "", name).strip()
    return name.lower()


# --- the pipeline ------------------------------------------------------------

class Pipeline:
    """Everything wired together. app.py builds one at startup."""

    def __init__(self, index, surveillance, references, embed, chat,
                 web_search=None, publications=None, scoping=None):
        self.index = index
        self.surveillance = surveillance
        self.references = references
        self.embed = embed
        self.chat = chat
        self.web_search = web_search
        self.publications = publications
        self.scoping = scoping

    # -- stages ---------------------------------------------------------------

    def route(self, message, history, country, answer):
        context = ""
        if history:
            lines = [f"{h['role']}: {h['content'][:200]}" for h in history[-4:]]
            context = "RECENT CONVERSATION:\n" + "\n".join(lines) + "\n\n"
        if country and country != "All Europe":
            context += f"USER'S SELECTED COUNTRY: {country}\n\n"

        started = time.time()
        try:
            completion = self.chat(
                model=ROUTER_MODEL,
                messages=[{"role": "system", "content": prompts.ROUTER},
                          {"role": "user", "content": f"{context}CURRENT USER MESSAGE: {message}"}],
                temperature=0, max_tokens=150)
            result, parsed = parse_router(completion.text, message)
            answer.usage["router"] = completion.usage
        except Exception as err:
            print(f"[ROUTER] failed, passing through as in-scope domain: {err}")
            result, parsed = {"rewritten": message, "in_scope": True, "route": "domain"}, False

        answer.timing["router"] = round(time.time() - started, 2)
        answer.router_parsed = parsed
        return result

    def gather(self, query, route, country, answer):
        """Evidence, web text, surveillance, publications — all of it.

        Only the first two depend on the route.
        """
        started = time.time()
        try:
            q_vec = self.embed([query])[0]
        except Exception as err:
            raise PipelineError(f"could not embed the query: {err}") from err
        answer.timing["embedding"] = round(time.time() - started, 2)

        if route in ("domain", "both"):
            started = time.time()
            top_k = TOP_K_BOTH if route == "both" else TOP_K_DOMAIN
            answer.hits = self.index.search(query, q_vec, top_k=top_k)
            answer.references = retrieval.references_for(answer.hits, self.references)
            answer.timing["retrieval"] = round(time.time() - started, 2)
            if answer.hits and max(h.cosine for h in answer.hits) < THIN_COVERAGE_COSINE:
                answer.thin_coverage = True

        if route in ("web", "both"):
            if self.web_search:
                started = time.time()
                try:
                    answer.web_text = self.web_search(query) or ""
                except Exception as err:
                    print(f"[WEB] search failed: {err}")
                    answer.web_text = ""
                answer.timing["web_search"] = round(time.time() - started, 2)
            # Set whether or not a search was attempted. If the agent failed to
            # start, the model has no current information and must be told so,
            # otherwise it fills the gap from memory.
            answer.web_failed = not answer.web_text.strip()

        # Independent of route from here down.
        chunk_text = " ".join(h.chunk.text for h in answer.hits)
        answer.diseases = ecdc.scan_text(query, chunk_text, answer.web_text)
        if answer.diseases and country and country != "All Europe":
            answer.ecdc_block = self.surveillance.block(answer.diseases, country)

        if self.publications:
            answer.publications = self.publications.match(q_vec, limit=2)
        if self.scoping:
            answer.scoping = self.scoping.match(q_vec, limit=1)

        return q_vec

    def generate(self, message, history, context, audience, country, length, answer):
        system = prompts.system_prompt(audience, country=country, length=length)
        messages = [{"role": "system", "content": system}] + list(history)
        messages.append({"role": "user",
                         "content": f"{context}\n\nQUESTION: {message}" if context else message})

        budget = LENGTH_TOKENS.get(length, LENGTH_TOKENS["Standard"]) + REASONING_HEADROOM
        started = time.time()
        try:
            completion = self.chat(model=GENERATION_MODEL, messages=messages,
                                   temperature=0.3, max_tokens=budget,
                                   reasoning_effort="high")
        except Exception as err:
            raise PipelineError(f"generation failed: {err}") from err
        answer.timing["generation"] = round(time.time() - started, 2)
        answer.usage["generation"] = completion.usage
        return completion

    # -- the whole thing ------------------------------------------------------

    def respond(self, message, history=None, audience="Policymaker",
                country="All Europe", length="Standard"):
        started = time.time()
        answer = Answer()

        if not message or not message.strip():
            answer.route = "empty"
            return answer

        # Reloads only when the data is a month old, or after a failed load.
        # Cheap on every other call, and the only thing keeping the monthly
        # extraction connected to a Space that may stay awake for months.
        self.surveillance.refresh_if_stale()

        history = clean_history(history)
        routing = self.route(message, history, country, answer)
        answer.rewritten = routing["rewritten"]
        answer.route = routing["route"]
        answer.in_scope = routing["in_scope"]

        # Every refusal comes from here. When the model produced the redirect
        # itself, the log recorded route: domain with a passage count and a word
        # count, indistinguishable from a real answer, so refusals could not be
        # counted.
        if not answer.in_scope and answer.route != "meta":
            answer.route = "redirect"
            answer.text = prompts.REDIRECT
            answer.timing["total"] = round(time.time() - started, 2)
            return answer

        if answer.route != "meta":
            self.gather(answer.rewritten, answer.route, country, answer)

        context = build_context(
            answer.hits, answer.references, answer.web_text, answer.ecdc_block,
            answer.publications, answer.scoping,
            web_failed=answer.web_failed, thin_coverage=answer.thin_coverage)
        answer.context_chars = len(context)

        completion = self.generate(message, history, context, audience, country, length, answer)

        text = postprocess.clean(completion.text, audience)
        text = self.append_further_reading(text, answer.publications)
        answer.text = text
        answer.violations = postprocess.check(
            text, audience=audience, length=length, had_context=bool(context))

        answer.timing["total"] = round(time.time() - started, 2)
        return answer

    @staticmethod
    def append_further_reading(text, publications):
        """Add a link for a publication the model used but did not cite."""
        if not publications:
            return text
        body, _, references = text.lower().partition("references")
        added = []
        for record, _ in publications:
            key = author_key(record["short"])
            if key and key in body and key not in references:
                added.append(record)
        if not added:
            return text
        lines = ["\n\nFurther reading from the IDAlert project:"]
        lines += [f"{r['short']}. {r['title']}. {r['link']}" for r in added]
        return text + "\n".join(lines)
