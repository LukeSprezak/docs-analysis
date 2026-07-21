from app.identity.domain.models import User
from app.identity.domain.repositories import UserRepo
from app.identity.security import verify_password
from app.shared.exceptions import AuthenticationException


class AuthenticateUserUseCase:
    def __init__(self, user_repo: UserRepo):
        self.user_repo = user_repo

    async def execute(self, email: str, password: str) -> User:
        normalized_email = email.strip().lower()
        user = await self.user_repo.get_by_email(normalized_email)
        # Ten sam komunikat dla braku użytkownika i złego hasła — nie zdradzamy,
        # czy dany email istnieje (ochrona przed enumeracją kont).
        if user is None or not verify_password(password, user.hashed_password):
            raise AuthenticationException("Invalid email or password")
        return user