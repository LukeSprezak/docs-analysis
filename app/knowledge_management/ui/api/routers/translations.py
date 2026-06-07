from fastapi import APIRouter

from app.shared.enums import LangProvider
from app.shared.translations import translations
from app.shared.exceptions import EntityNotFoundException

router = APIRouter(tags=["translations"])

@router.get("/translations/{lang}")
async def get_translations(lang: str):
    if lang not in translations:
        if LangProvider.ENGLISH in translations:
            return translations[LangProvider.ENGLISH]
        raise EntityNotFoundException(entity="Language", identifier=lang)
    return translations[lang]
