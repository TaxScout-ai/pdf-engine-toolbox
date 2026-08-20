"""Custom exception classes for the PDF Engine."""


class PdfEngineError(Exception):
    """Base exception for PDF engine errors."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class PdfCorruptError(PdfEngineError):
    """PDF file is corrupt or unreadable."""

    def __init__(self, message: str = "PDF file is corrupt or cannot be opened"):
        super().__init__("PDF_CORRUPT", message, 422)


class PageOutOfRangeError(PdfEngineError):
    """Page index is out of range."""

    def __init__(self, page: int, total: int):
        super().__init__(
            "PAGE_OUT_OF_RANGE",
            f"Page {page} is out of range (document has {total} pages)",
            422,
        )


class DownloadFailedError(PdfEngineError):
    """Failed to download PDF from presigned URL."""

    def __init__(self, message: str = "Failed to download PDF from source URL"):
        super().__init__("DOWNLOAD_FAILED", message, 502)


class AuthenticationError(PdfEngineError):
    """Request authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__("AUTH_FAILED", message, 401)


class PdfPasswordAuthorizationError(PdfEngineError):
    """Known-password authorization failed without exposing an auth oracle."""

    def __init__(self):
        super().__init__(
            "PDF_PASSWORD_AUTHORIZATION_FAILED",
            "Password is incorrect or document access is not authorized",
            403,
        )


class PdfPasswordNotRequiredError(PdfEngineError):
    """The caller attempted password authorization for a plain PDF."""

    def __init__(self):
        super().__init__(
            "PDF_PASSWORD_NOT_REQUIRED",
            "Document does not require password authorization",
            409,
        )


class TimeoutError(PdfEngineError):
    """Operation timed out."""

    def __init__(self, message: str = "Operation timed out"):
        super().__init__("TIMEOUT", message, 504)
