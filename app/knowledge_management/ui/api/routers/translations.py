from fastapi import APIRouter

from app.shared.exceptions import EntityNotFoundException
from app.shared.translations import translations

router = APIRouter(tags=["translations"])


@router.get("/translations/{lang}")
async def get_translations(lang: str) -> dict[str, str]:
    if lang not in translations:
        # Fallback to English if language not found
        if "en" in translations:
            return translations["en"]
        raise EntityNotFoundException(entity="Language", identifier=lang)
    return translations[lang]
