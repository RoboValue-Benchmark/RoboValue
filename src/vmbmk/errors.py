class VMBMKError(Exception):
    """Base class for actionable benchmark errors."""


class ConfigurationError(VMBMKError):
    """Raised when a registry or plugin configuration is invalid."""


class DataValidationError(VMBMKError):
    """Raised when task data violates the VMBMK format."""
