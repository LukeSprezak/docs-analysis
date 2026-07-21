"""Tests for the pure logic of BasePostgresRepo — no network, no database.

The connection operations (`_execute_statement`/`_fetch_*`) need a live Postgres and are out of
scope for an offline test; here we cover the deterministic JSONB helper that decides whether a
column has to be deserialized.
"""

from app.shared.postgres_repo import BasePostgresRepo


def test_deserializes_jsonb_returned_as_text() -> None:
    assert BasePostgresRepo._deserialize_json_column('{"a": 1}') == {"a": 1}
    assert BasePostgresRepo._deserialize_json_column("[1, 2, 3]") == [1, 2, 3]


def test_returns_dict_and_list_unchanged() -> None:
    already_dict = {"owner_id": "u1"}
    already_list = ["doc-1", "doc-2"]
    assert BasePostgresRepo._deserialize_json_column(already_dict) is already_dict
    assert BasePostgresRepo._deserialize_json_column(already_list) is already_list


def test_returns_none_without_attempting_to_parse() -> None:
    assert BasePostgresRepo._deserialize_json_column(None) is None
