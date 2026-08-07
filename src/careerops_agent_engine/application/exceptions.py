"""Controlled application-layer exceptions."""


class EvidenceDiscoveryValidationError(ValueError):
    """Evidence-agent output failed deterministic validation."""


class CVProposalValidationError(ValueError):
    """Generated CV proposal failed deterministic validation."""


class CVClaimVerificationValidationError(ValueError):
    """Claim-verification output failed deterministic validation."""
