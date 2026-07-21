from dataclasses import dataclass


@dataclass
class User:
    id: str
    email: str
    hashed_password: str
    created_at: str | None = None
