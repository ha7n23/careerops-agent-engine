"""Controlled application-layer exceptions."""


class EvidenceDiscoveryValidationError(ValueError):
    """Evidence-agent output failed deterministic validation."""


class CareerEvidenceUnavailableError(ValueError):
    """An approved evidence record cannot be accessed by the user."""


class CareerEvidenceEditValidationError(ValueError):
    """An approved evidence edit failed deterministic validation."""


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


class StructuredCVAssemblyError(ValueError):
    """Parsed source CV cannot be assembled without losing content."""


class CVRenderingError(ValueError):
    """A structured CV could not be rendered safely."""


class CVArtifactVerificationError(ValueError):
    """A generated CV artifact could not enter verification safely."""


class CVPDFConversionError(ValueError):
    """A verified DOCX could not be converted safely to PDF."""


class FinalCVGenerationError(ValueError):
    """A final CV version could not complete its trusted artifact workflow."""


class CVArtifactRetrievalError(ValueError):
    """A generated CV version or artifact could not be retrieved safely."""
