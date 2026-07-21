from app.identity.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_is_not_plaintext_and_verifies():
    hashed = hash_password("secret-password")
    assert hashed != "secret-password"
    assert verify_password("secret-password", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("secret-password")
    assert verify_password("other-password", hashed) is False


def test_verify_password_fail_closed_on_corrupted_hash():
    assert verify_password("whatever", "this-is-not-bcrypt") is False


def test_token_roundtrip_returns_subject():
    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_decode_rejects_garbage_token():
    assert decode_access_token("not.a.token") is None


def test_decode_rejects_token_signed_with_other_secret():
    import jwt

    foreign = jwt.encode({"sub": "user-123"}, "another-secret", algorithm="HS256")
    assert decode_access_token(foreign) is None
