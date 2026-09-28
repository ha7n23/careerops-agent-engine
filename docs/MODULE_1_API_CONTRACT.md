# CareerOps Module 1 API Contract — v1

[← Back to README](../README.md) · [Architecture](ARCHITECTURE.md)

Status: **frozen for Module 2 and Module 3 integration as of 2026-09-28**.

This document defines the stable service boundary exposed by the CareerOps
Agent Engine. FastAPI's generated OpenAPI document at `/openapi.json` remains
the canonical field-level schema; this document freezes the workflow,
authentication, lifecycle, retry, and error semantics that callers depend on.

## 1. Compatibility policy

- Existing `/api/v1` request fields, response fields, enum values, and
  behaviours must not be removed or reinterpreted.
- Optional response fields and new endpoints may be added compatibly.
- Breaking changes require a new versioned route or an explicit coordinated
  migration across Modules 1, 2, and 3.
- Opaque identifiers must be treated as strings. Callers must not derive
  meaning from their prefixes.
- Timestamps returned by the API are ISO 8601 values.

## 2. Base URL and authentication

Local base URL: `http://127.0.0.1:8000`.

`GET /health`, `GET /ready`, `/docs`, and `/openapi.json` are public. Every
business endpoint requires:

```http
X-User-ID: USER-001
```

The value must match `^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$`. When
`CAREEROPS_AUTH_MODE=service_key`, callers must also send:

```http
X-CareerOps-Service-Key: <configured service credential>
```

Module 2 owns trusted service-to-service authentication. Module 3 must call
Module 1 through Module 2 rather than exposing the service key to a browser.

All business reads and writes are scoped by `X-User-ID`. Unknown identifiers
and identifiers owned by another user intentionally produce the same opaque
`404` response.

## 3. Endpoint inventory

| Method | Path | Success | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | `200` | Process liveness |
| `GET` | `/ready` | `200` | Database/schema readiness |
| `GET` | `/api/v1/cv-documents` | `200` | List source-document history |
| `POST` | `/api/v1/cv-documents` | `201` | Upload a PDF or DOCX source |
| `POST` | `/api/v1/cv-documents/text` | `201` | Create a pasted-text source |
| `POST` | `/api/v1/cv-documents/{document_id}/evidence-review` | `200` | Start or recover evidence extraction/review |
| `GET` | `/api/v1/cv-evidence-reviews` | `200` | List evidence-review history |
| `GET` | `/api/v1/cv-evidence-reviews/{review_run_id}` | `200` | Recover one evidence-review run |
| `POST` | `/api/v1/cv-evidence-reviews/{review_run_id}/review` | `200` | Submit an evidence decision |
| `GET` | `/api/v1/evidence` | `200` | Search/filter/page approved evidence |
| `GET` | `/api/v1/evidence/{evidence_id}` | `200` | Retrieve approved evidence |
| `PATCH` | `/api/v1/evidence/{evidence_id}` | `200` | Edit grounded evidence fields |
| `POST` | `/api/v1/evidence/{evidence_id}/archive` | `200` | Archive evidence idempotently |
| `POST` | `/api/v1/evidence/{evidence_id}/restore` | `200` | Restore evidence idempotently |
| `POST` | `/api/v1/job-analysis` | `200` | Start durable job analysis |
| `GET` | `/api/v1/job-analysis/{thread_id}` | `200` | Recover durable job analysis |
| `POST` | `/api/v1/job-analysis/{thread_id}/review` | `200` | Resume paused CV review |
| `POST` | `/api/v1/cv-versions` | `200` | Generate or recover a final CV |
| `GET` | `/api/v1/cv-versions/{cv_version_id}` | `200` | Retrieve safe version metadata |
| `GET` | `/api/v1/cv-versions/{cv_version_id}/artifacts/{artifact_format}` | `200` | Download a verified artifact |

## 4. Evidence-source ingestion

### File upload

`POST /api/v1/cv-documents` accepts `multipart/form-data` with one `file`
field. Supported formats are byte-validated PDF and DOCX files, up to the
configured upload limit (5 MiB by default).

### Pasted text

`POST /api/v1/cv-documents/text` accepts:

```json
{
  "title": "Career history notes",
  "content": "Built a FastAPI service using Python."
}
```

`title` and `content` must be non-empty. Content is limited to 30,000
characters by default. It is stored as a trusted UTF-8 source with
`manual_entry` provenance; it does not become approved evidence immediately.

Both ingestion endpoints return safe document metadata:

```json
{
  "document_id": "DOC-...",
  "original_filename": "Career history notes.txt",
  "document_format": "text",
  "media_type": "text/plain; charset=utf-8",
  "size_bytes": 39,
  "sha256_hex": "...",
  "status": "uploaded"
}
```

Document history accepts `limit=1..100`, is newest-first, and returns
`items`, `count`, and `limit` without storage paths or owning user IDs.

## 5. Evidence extraction and human review

Call
`POST /api/v1/cv-documents/{document_id}/evidence-review` after ingestion.
The response contains:

- `review_run_id`, `document_id`, and `status`;
- grounded `proposals`;
- deterministic `overlap_findings`;
- `document_warnings`;
- `review_result` after completion, otherwise `null`.

Statuses are `awaiting_review`, `completed`, or `invalid`. Starting the same
document again recovers its existing run rather than creating another one.

Submit a complete decision to
`POST /api/v1/cv-evidence-reviews/{review_run_id}/review`:

```json
{
  "approved_proposal_ids": ["EVP-001"],
  "rejected_proposal_ids": [],
  "edits": [],
  "duplicate_resolutions": [],
  "reviewer_comment": "Checked against the source."
}
```

Every proposal must be classified exactly once by approval, rejection, or an
edit. Exact retries of a completed decision return the persisted result and do
not append another audit record.

Each reported overlap requires an explicit resolution. Supported actions are:

| Scope | Actions | Required target |
| --- | --- | --- |
| `within_document` | `keep_existing`, `accept_separate` | `matching_proposal_id` |
| `approved_evidence` | `keep_existing`, `accept_separate`, `replace_existing`, `merge_into_existing` | `matching_evidence_id` |

Example merge resolution:

```json
{
  "proposal_id": "EVP-001",
  "scope": "approved_evidence",
  "action": "merge_into_existing",
  "matching_proposal_id": null,
  "matching_evidence_id": "EVD-001"
}
```

Review-history listing accepts `limit=1..100`, is newest-first, and exposes
summary counts rather than complete private review payloads.

## 6. Approved Evidence Registry

Only human-approved records enter the registry. The public evidence shape is:

```json
{
  "evidence_id": "EVD-001",
  "category": "project",
  "title": "CareerOps Agent Engine",
  "verification_status": "approved",
  "lifecycle_status": "active",
  "technologies": ["Python", "FastAPI"],
  "capabilities": ["API development"],
  "approved_claims": ["Built a FastAPI service using Python."],
  "source_references": []
}
```

### Search, filtering, and pagination

`GET /api/v1/evidence` supports:

| Parameter | Contract |
| --- | --- |
| `q` | Optional, 1–200 characters; all normalized tokens must match across searchable fields |
| `category` | Optional: `project`, `employment`, `education`, `certification`, `achievement`, `skill` |
| `lifecycle_status` | `active` by default; `archived` only when explicitly requested |
| `offset` | `0..10000`, default `0` |
| `limit` | `1..100`, default `100` |

The response contains `items`, page `count`, matching `total`, `offset`,
`limit`, and `has_more`. Ordering is stable by normalized title and evidence
identifier.

### Editing and lifecycle

`PATCH /api/v1/evidence/{evidence_id}` accepts at least one of `category`,
`title`, `technologies`, `capabilities`, or `approved_claims`. Ownership,
verification state, identity, and source references are immutable. Newly added
technologies and claims must remain grounded in trusted source excerpts.

Archive and restore are idempotent. Archived evidence:

- remains retrievable and auditable through the management API;
- appears only when `lifecycle_status=archived` is requested;
- is excluded from default registry results, evidence discovery, fit scoring,
  and job-analysis grounding.

Restored evidence becomes active and immediately eligible for downstream job
analysis again. Edits, archive operations, and restore operations append
before/after lifecycle audit records.

## 7. Durable job analysis

Start with `POST /api/v1/job-analysis`:

```json
{
  "job_id": "JOB-001",
  "job_description": "Junior AI Engineer requiring Python and FastAPI."
}
```

The response is a discriminated union:

- `status=awaiting_review` includes `review` with proposals, verification
  reports, and allowed actions;
- `status=completed` includes `review_status` and `final_cv_proposals`.

Both forms include `thread_id`, extracted requirements, evidence matches,
deterministic `fit_score`, generated proposals, verification reports,
reviewable/blocked proposal IDs, and lightweight audit events.

Use `GET /api/v1/job-analysis/{thread_id}` to recover the durable state. Resume
an awaiting-review thread with:

```json
{
  "action": "approve",
  "approved_proposal_ids": ["CVP-001"],
  "rejected_proposal_ids": [],
  "edits": [],
  "reviewer_comment": null
}
```

Actions are `approve`, `edit`, `reject`, and `regenerate`. Decisions must only
reference proposals exposed by that thread. Human-edited and regenerated text
is re-verified before it can become final.

## 8. Final CV versions and artifacts

Generate from an accepted job-analysis thread and its source document:

```json
{
  "thread_id": "THR-001",
  "source_document_id": "DOC-001"
}
```

`POST /api/v1/cv-versions` returns immutable version metadata, verified DOCX
and PDF artifact metadata, and `reused_existing_version`. Retrying the same
accepted workflow/document pair recovers the existing version rather than
silently creating a duplicate.

Retrieve metadata with `GET /api/v1/cv-versions/{cv_version_id}`. Download only
verified artifacts with `artifact_format=docx` or `artifact_format=pdf`.
Downloads return attachment bytes and never expose private storage keys.

## 9. Error contract

FastAPI validation errors use the standard `detail` response. Application
errors use `{"detail": "..."}` with these stable status meanings:

| Status | Meaning |
| --- | --- |
| `401` | Missing/invalid user scope or service credential |
| `404` | Resource unavailable, including cross-user identifiers |
| `409` | State conflict or final-CV prerequisites not satisfied |
| `422` | Invalid input, decision, edit, extraction, or workflow request |
| `500` | Sanitized unexpected internal failure |
| `502` | Invalid LLM structured result or document-rendering/conversion failure |

Callers may display safe `4xx` details. They should treat `5xx` details as
operational messages, avoid automatic mutation retries, and retain the opaque
workflow identifier for recovery.

## 10. Proven contract properties

The frozen contract is covered by deterministic and opt-in live integration
proofs:

- ingestion → proposals → duplicate resolution → approval → registry;
- exact evidence-review retry idempotency;
- archive exclusion and restore reactivation in job analysis;
- cross-user isolation across retrieval, queries, and discovery;
- PostgreSQL/Alembic initialization and non-root container health;
- real Groq job-analysis execution through the safety boundary;
- real HTTP DOCX/PDF generation, verification, download, retry, and isolation.

