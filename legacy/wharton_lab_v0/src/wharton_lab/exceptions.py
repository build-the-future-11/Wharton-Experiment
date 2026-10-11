"""Lab-specific errors."""


class BlockedClientError(RuntimeError):
    """Raised when client-specific inputs (e.g. liabilities) are required but missing."""

    def __init__(self, message: str = "Client liability schedule required but not provided."):
        super().__init__(message)
