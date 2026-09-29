from typing import Any


class AppException(Exception):
    """Base class for all application exceptions."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_SERVER_ERROR",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.context = context or {}


class EntityNotFoundException(AppException):
    def __init__(self, entity: str, identifier: Any, context: dict[str, Any] | None = None):
        message = f"{entity} with identifier {identifier} not found"
        super().__init__(
            message=message,
            status_code=404,
            error_code="ENTITY_NOT_FOUND",
            context={"entity": entity, "identifier": identifier, **(context or {})},
        )


class ValidationException(AppException):
    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(
            message=message, status_code=400, error_code="VALIDATION_ERROR", context=context
        )


class AuthenticationException(AppException):
    def __init__(
        self, message: str = "Authentication required", context: dict[str, Any] | None = None
    ):
        super().__init__(
            message=message, status_code=401, error_code="AUTHENTICATION_ERROR", context=context
        )


class EmbeddingModelMismatchException(AppException):
    def __init__(self, indexed_with: str, configured: str):
        super().__init__(
            message=(
                f"The vector index was built with embedding model '{indexed_with}', but "
                f"'{configured}' is configured. Switch back to '{indexed_with}', or delete all "
                "documents and upload them again."
            ),
            status_code=500,
            error_code="EMBEDDING_MODEL_MISMATCH",
            context={"indexed_with": indexed_with, "configured": configured},
        )


class InfrastructureException(AppException):
    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(
            message=message, status_code=500, error_code="INFRASTRUCTURE_ERROR", context=context
        )
