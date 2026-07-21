"""Współdzielony limiter zapytań (slowapi).

Klucz limitu = adres IP klienta (`get_remote_address`). Globalny limit domyślny działa
jako bezpiecznik dla wszystkich endpointów, a stuningowane limity na drogich endpointach
(LLM, upload, auth) nakładane są dekoratorem `@limiter.limit(...)` w routerach.

Limiter jest jednym, współdzielonym obiektem — endpointy importują właśnie ten `limiter`,
a w testach jest wyłączany (`limiter.enabled = False`), żeby powtarzane wywołania z tego
samego adresu nie wpadały w limit.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.shared.config import settings

# headers_enabled celowo False: wstrzykiwanie nagłówków X-RateLimit-* wymagałoby
# parametru `response: Response` w każdym chronionym endpoincie. Sam limit działa bez tego,
# a przekroczenie zwraca spójny błąd 429 (rate_limit_exceeded_handler).
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    enabled=settings.RATE_LIMIT_ENABLED,
)
