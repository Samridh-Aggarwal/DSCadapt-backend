---
title: DSCAdapt API
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DSCAdapt — Climate-Health Decision Support API

Backend for [DSCAdapt](https://samridh-aggarwal.github.io/dscadapt/), a decision-support
tool on how climate adaptation measures affect infectious disease risk. Built at the
Grantham Research Institute, LSE, under the IDAlert project (Horizon Europe, grant
101057554).

**Deployed automatically from [Samridh-Aggarwal/dscadapt](https://github.com/Samridh-Aggarwal/dscadapt).**
Do not edit here — anything changed in this Space is overwritten on the next push to
`backend/` in that repository.

## Endpoints

| route | what it does |
|---|---|
| `GET /` | health, including whether search, translation and surveillance data are working |
| `POST /ask` | the pipeline: route, retrieve, generate, attribute |
| `POST /transcript` | a session as a PDF |
| `POST /briefing` | a synthesised, audited briefing as a PDF |
| `POST /evaluate` | review an uploaded policy document for overlooked disease risks |
| `POST /extract` | text out of a PDF, DOCX or TXT |

`GET /debug/graph` and `GET /debug/corpus` exist when `EXPOSE_DEBUG_ROUTES=1`.

## Configuration

| variable | required | default |
|---|---|---|
| `MISTRAL_API_KEY` | yes | — |
| `DEEPL_API_KEY` | no | translation disabled without it |
| `HF_TOKEN` | no | needed only to upload traces |
| `ECDC_DATASET` | no | `Samridh25/ecdc-surveillance` |
| `LOGS_DATASET` | no | unset means traces stay local |
| `LOG_QUESTION_TEXT` | no | `true` |
| `API_KEY` | no | unset means no authentication |
| `RATE_LIMIT` | no | `0`, meaning off |
| `ALLOW_ORIGINS` | no | `*` |
| `EXPOSE_DEBUG_ROUTES` | no | `false` |

## Logging

One JSON line per query, written locally and pushed to `LOGS_DATASET` in the background
if it is set. A trace holds the question, the answer, the settings, the retrieved
sections with their scores, and any output-rule violations. Set `LOG_QUESTION_TEXT=false`
to record lengths instead of text.

Uploaded documents are never logged — only a length and a hash.
