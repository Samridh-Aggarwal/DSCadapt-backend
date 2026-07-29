"""Read the environment, build the clients, wire everything together.

app.py should read as a list of routes, so the assembly lives here. Everything
external — keys, repository names, model names — comes from the environment
with a documented default, because the alternative is what the old code did:
the logs repository was a constant naming one person's private dataset, so
anyone forking this and running it with their own token would have written
their traffic into someone else's repo.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import attribution
import corpus
import ecdc
import logger as logging_
import ontology
import pipeline
import prompts
import retrieval
import review
import translate

HERE = Path(__file__).parent


def env(name, default=None, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        print(f"[CONFIG] {name}={raw!r} is not a valid {cast.__name__}, using {default!r}")
        return default


@dataclass
class Settings:
    mistral_key: str = field(default_factory=lambda: env("MISTRAL_API_KEY"))
    deepl_key: str = field(default_factory=lambda: env("DEEPL_API_KEY"))
    hf_token: str = field(default_factory=lambda: env("HF_TOKEN"))

    ecdc_repo: str = field(default_factory=lambda: env("ECDC_DATASET", "Samridh25/ecdc-surveillance"))
    ecdc_file: str = field(default_factory=lambda: env("ECDC_FILE", "ecdc_cases.csv"))

    logs_repo: str = field(default_factory=lambda: env("LOGS_DATASET"))
    logs_dir: str = field(default_factory=lambda: env("LOGS_DIR", "/tmp/dscadapt-logs"))
    log_text: bool = field(default_factory=lambda: env("LOG_QUESTION_TEXT", True, _boolean))
    log_echo: bool = field(default_factory=lambda: env("LOG_ECHO", True, _boolean))

    docs_dir: str = field(default_factory=lambda: env("DOCS_DIR", str(HERE / "docs")))
    allow_origins: str = field(default_factory=lambda: env("ALLOW_ORIGINS", "*"))

    def summary(self):
        return {
            "mistral": bool(self.mistral_key),
            "deepl": bool(self.deepl_key),
            "ecdc_dataset": self.ecdc_repo,
            "logs_dataset": self.logs_repo or "local only",
            "log_question_text": self.log_text,
        }


def _boolean(raw):
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# --- clients -----------------------------------------------------------------

@dataclass
class Completion:
    """What every chat call returns, whichever provider made it."""
    text: str
    usage: dict = None
    raw: object = None


def build_mistral(api_key):
    from mistralai.client import Mistral
    client = Mistral(api_key=api_key)
    try:
        slow = Mistral(api_key=api_key, timeout_ms=90_000)
    except TypeError:
        slow = client
    return client, slow


def make_chat(client):
    """A callable the pipeline can hold without knowing about Mistral."""
    def chat(model, messages, **kwargs):
        response = client.chat.complete(model=model, messages=messages, **kwargs)
        return Completion(text=_text_of(response), usage=_usage_of(response), raw=response)
    return chat


def make_embed(client):
    def embed(texts):
        response = client.embeddings.create(model="mistral-embed", inputs=texts)
        return [item.embedding for item in response.data]
    return embed


def _text_of(response):
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(block.text for block in content
                       if getattr(block, "type", "") == "text").strip()
    return (content or "").strip()


def _usage_of(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    return {"input": getattr(usage, "prompt_tokens", 0) or 0,
            "output": getattr(usage, "completion_tokens", 0) or 0}


WEB_INSTRUCTIONS = (
    "Search for current information directly relevant to infectious disease risk for the "
    "location and measures mentioned: disease surveillance, outbreaks, vector or pathogen "
    "trends, and public health or vector-control measures and regulations. Include local "
    "climate or geographic context only where it bears on disease risk. Do not return general "
    "climate, emissions, energy or ESG policy that is not specific to infectious disease or "
    "public health. Describe what measures, programmes or regulations exist and what they do "
    "in plain terms, but do not state specific regulation, directive or decree numbers, "
    "article numbers, exact enactment dates or precise statistics as fact, because web sources "
    "are frequently wrong on those identifiers even when the substance is right. Report the "
    "substance and include the source links so the exact details can be verified.")

import re  # noqa: E402  (used by the law-number filter below)

LAW_NUMBER = re.compile(
    r"\((?:EU|EC|EEC)\)\s*(?:No\.?\s*)?\d{1,4}/\d{2,4}"
    r"|(?:Royal Decree(?:-Law)?|Real Decreto|Decreto)\s+\d+/\d{4}", re.IGNORECASE)

LAW_PLACEHOLDER = "[official number removed — verify on EUR-Lex or the national gazette]"


def make_web_search(client, agent_id):
    """Search, then strip identifiers before the generation model ever sees them.

    Web results are usually right on substance and often wrong on the exact
    number, and a wrong directive number in a policy brief is worse than no
    number at all. Removing them here rather than asking the model not to
    repeat them is the difference between a rule that can fail and one that
    cannot.
    """
    if agent_id is None:
        return None

    def search(query):
        response = client.beta.conversations.start(agent_id=agent_id, inputs=query)
        texts, sources, searched = [], [], False
        for entry in response.outputs:
            if getattr(entry, "type", "") == "tool.execution":
                searched = True
            if getattr(entry, "type", "") != "message.output":
                continue
            for block in getattr(entry, "content", []) or []:
                kind = getattr(block, "type", "")
                if kind == "text":
                    texts.append(block.text)
                elif kind == "tool_reference":
                    title, url = getattr(block, "title", ""), getattr(block, "url", "")
                    if title or url:
                        sources.append(f"{title}: {url}")
        if not searched:
            print("[WEB] the agent did not search; discarding ungrounded text")
            return ""
        result = "\n".join(texts)
        if sources:
            result += "\n\nWeb sources:\n" + "\n".join(sources[:5])
        return LAW_NUMBER.sub(LAW_PLACEHOLDER, result)

    return search


def create_web_agent(client, model="mistral-medium-latest"):
    try:
        agent = client.beta.agents.create(
            model=model, name="DSCAdapt Web Search",
            description="Searches the web for current policy, legislation and geographic context.",
            instructions=WEB_INSTRUCTIONS, tools=[{"type": "web_search"}],
            completion_args={"temperature": 0.2})
        print("[WEB] search agent ready")
        return agent.id
    except Exception as err:
        print(f"[WEB] search not available: {err}")
        return None


# --- the assembled system ----------------------------------------------------

@dataclass
class Services:
    settings: Settings
    corpus: object
    index: object
    surveillance: object
    pipeline: object
    reviewer: object
    translator: object
    log: object
    embed: object
    chat: object
    web_agent_id: str = None

    def health(self):
        return {
            "status": "ok",
            "model": pipeline.GENERATION_MODEL,
            "chunks": len(self.index),
            "documents": len({c.source for c in self.corpus.chunks}),
            "web_search": self.web_agent_id is not None,
            "deepl": self.translator.available,
            "ecdc": self.surveillance.available,
            "ecdc_years": self.surveillance.year_range,
            "corpus_problems": len(self.corpus.problems),
            "version": self.log.version,
        }

    @classmethod
    def build(cls, settings=None):
        settings = settings or Settings()
        print(f"[CONFIG] {settings.summary()}")

        client, slow_client = build_mistral(settings.mistral_key)
        chat, embed = make_chat(client), make_embed(client)
        agent_id = create_web_agent(client)

        known = {m["name"] for m in ontology.MEASURES}
        loaded = corpus.load(root=settings.docs_dir, known_measures=known)
        print(loaded.report())

        index = retrieval.Index.build(loaded.indexed(), embed)
        print(f"[CORPUS] {len(index)} chunks indexed of {len(loaded.chunks)} parsed")

        publications = retrieval.Matcher.build(
            loaded.publications,
            [f"{p['title']}. {p.get('text','')}" for p in loaded.publications],
            embed, threshold=0.84)
        scoping_chunks = loaded.of_kind("scoping_review")
        scoping = retrieval.Matcher.build(
            [{"section": c.section, "text": c.text} for c in scoping_chunks],
            [f"{c.section}. {c.text}" for c in scoping_chunks],
            embed, threshold=0.78)

        surveillance = ecdc.Surveillance.from_hub(settings.ecdc_repo, settings.ecdc_file)

        sink = (logging_.HuggingFaceSink(settings.logs_repo, settings.hf_token)
                if settings.logs_repo and settings.hf_token else logging_.NullSink())
        if isinstance(sink, logging_.NullSink):
            print("[LOGGER] no LOGS_DATASET set, keeping traces local only")

        return cls(
            settings=settings, corpus=loaded, index=index, surveillance=surveillance,
            pipeline=pipeline.Pipeline(
                index=index, surveillance=surveillance, references=loaded.references,
                embed=embed, chat=chat,
                web_search=make_web_search(slow_client, agent_id),
                publications=publications, scoping=scoping),
            reviewer=review.Reviewer(index, surveillance, loaded.measures, embed, chat),
            translator=translate.Translator.from_key(settings.deepl_key),
            log=logging_.Logger(directory=settings.logs_dir, sink=sink,
                                echo=settings.log_echo),
            embed=embed, chat=chat, web_agent_id=agent_id)
