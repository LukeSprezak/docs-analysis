from abc import ABC, abstractmethod

from .models import User


class UserRepo(ABC):
    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        pass

    @abstractmethod
    async def get_by_id(self, user_id: str) -> User | None:
        pass

    @abstractmethod
    async def save(self, user: User) -> None:
        pass