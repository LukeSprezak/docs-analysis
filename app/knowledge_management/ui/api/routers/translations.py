from fastapi import APIRouter

from app.shared.translations import translations

router = APIRouter(tags=["translations"])


@router.get("/translations/{lang}")
async def get_translations(lang: str) -> dict[str, str]:
    # Every language is served from an in-memory dict that always contains
    # "en", so an unknown code falls back rather than failing.
    return translations.get(lang, translations["en"])
