"""Testy czystej logiki BasePostgresRepo — bez sieci/bazy.

Operacje na połączeniu (`_execute_statement`/`_fetch_*`) wymagają żywego Postgresa i są
poza zasięgiem testu offline; tu pokrywamy deterministyczny pomocnik na JSONB, który
decyduje, czy kolumnę trzeba zdeserializować.
"""

from app.shared.postgres_repo import BasePostgresRepo


def test_deserializuje_jsonb_zwrocony_jako_tekst() -> None:
    assert BasePostgresRepo._deserialize_json_column('{"a": 1}') == {"a": 1}
    assert BasePostgresRepo._deserialize_json_column("[1, 2, 3]") == [1, 2, 3]


def test_zwraca_dict_i_list_bez_zmian() -> None:
    already_dict = {"owner_id": "u1"}
    already_list = ["doc-1", "doc-2"]
    assert BasePostgresRepo._deserialize_json_column(already_dict) is already_dict
    assert BasePostgresRepo._deserialize_json_column(already_list) is already_list


def test_zwraca_none_bez_proby_parsowania() -> None:
    assert BasePostgresRepo._deserialize_json_column(None) is None