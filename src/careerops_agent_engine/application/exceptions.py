"""Controlled application-layer exceptions."""


class EvidenceDiscoveryValidationError(ValueError):
    """Evidence-agent output failed deterministic validation."""


class CVProposalValidationError(ValueError):
    """Generated CV proposal failed deterministic validation."""


class CVClaimVerificationValidationError(ValueError):
    """Claim-verification output failed deterministic validation."""


class CVReviewValidationError(ValueError):
    """Human CV-review input failed deterministic validation."""


class JobAnalysisThreadUnavailableError(ValueError):
    """A durable review thread cannot be accessed or resumed."""


class DocumentUploadValidationError(ValueError):
    """Uploaded career document failed deterministic validation."""


class DocumentExtractionError(ValueError):
    """Validated document could not be extracted safely."""


class DocumentTextUnavailableError(DocumentExtractionError):
    """Document contains no usable native text."""


class CVEvidenceProposalValidationError(ValueError):
    """CV evidence extraction failed deterministic validation."""


class CVEvidenceReviewValidationError(ValueError):
    """Human CV-evidence review failed deterministic validation."""


class CareerDocumentUnavailableError(ValueError):
    """A user-owned career document cannot be accessed for processing."""


class CVEvidenceReviewRunUnavailableError(ValueError):
    """A CV evidence-review run cannot be accessed or reviewed."""
