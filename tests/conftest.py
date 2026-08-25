"""Shared pytest configuration for isolated test settings."""

import os

# Tests must not inherit runtime settings from a developer's local .env.
# Explicit Settings(...) arguments in security tests still take precedence.
os.environ["CAREEROPS_ENVIRONMENT"] = "development"
os.environ["CAREEROPS_AUTH_MODE"] = "development"
