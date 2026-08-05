"""Enumerations shared across the CareerOps domain."""

from enum import StrEnum


class RequirementCategory(StrEnum):
    """How strongly a job description requires a capability."""

    ESSENTIAL = "essential"
    DESIRABLE = "desirable"


class MatchStrength(StrEnum):
    """Strength of the evidence supporting a job requirement."""

    STRONG = "strong"
    PARTIAL = "partial"
    RELATED = "related"
    NONE = "none"


class VerificationStatus(StrEnum):
    """Human-verification status of a career evidence record."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class EvidenceCategory(StrEnum):
    """High-level category of a career evidence record."""

    PROJECT = "project"
    EMPLOYMENT = "employment"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    ACHIEVEMENT = "achievement"
    SKILL = "skill"


class EvidenceSourceType(StrEnum):
    """Origin from which career evidence was obtained."""

    UPLOADED_CV = "uploaded_cv"
    MANUAL_ENTRY = "manual_entry"
    GITHUB = "github"
    EMPLOYMENT_RECORD = "employment_record"
    EDUCATION_RECORD = "education_record"
    CERTIFICATE = "certificate"


class CVSection(StrEnum):
    """Supported sections of a generated CV."""

    PROFILE = "profile"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"


class ReviewAction(StrEnum):
    """Action selected during human review."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    REGENERATE = "regenerate"


class ApprovalStatus(StrEnum):
    """Lifecycle status of a human-review request."""

    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    REGENERATION_REQUESTED = "regeneration_requested"
