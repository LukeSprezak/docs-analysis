import unicodedata

import pytest

from app.knowledge_management.domain.entity_normalization import normalize_entity_name


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Postgres", "postgres"),  # case
        ("Postgres", "  Postgres  "),  # surrounding whitespace
        ("Postgres", "Postgres."),  # trailing sentence punctuation
        ("Postgres", '"Postgres"'),  # quoting
        ("Apache Kafka", "Apache  Kafka"),  # collapsed inner whitespace
        ("Kafka", "(Kafka)"),  # brackets
    ],
)
def test_spelling_variants_share_one_key(first: str, second: str) -> None:
    assert normalize_entity_name(first) == normalize_entity_name(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Synonyms are NOT merged — that needs an alias list, not a string rule.
        ("Postgres", "PostgreSQL"),
        # The dangerous case a fuzzy matcher would get wrong: one character apart, different
        # things. Under-merging is recoverable; a silent bad merge fuses unrelated facts.
        ("Python 2", "Python 3"),
        # Diacritics are meaningful in Polish and are preserved.
        ("Łódź", "Lodz"),
    ],
)
def test_distinct_entities_keep_distinct_keys(first: str, second: str) -> None:
    assert normalize_entity_name(first) != normalize_entity_name(second)


def test_punctuation_only_names_do_not_collapse_into_one_key():
    # An empty key would merge every such name into a single node.
    assert normalize_entity_name("...") != normalize_entity_name("???")


def test_key_is_stable_across_unicode_composition_forms():
    # The same accented name arrives pre-composed (NFC) or decomposed (NFD) depending on the
    # source. They are different byte sequences, so without normalization they would become
    # two nodes that look identical in every UI.
    composed = unicodedata.normalize("NFC", "Łódź")
    decomposed = unicodedata.normalize("NFD", "Łódź")
    assert composed != decomposed  # genuinely different strings
    assert normalize_entity_name(composed) == normalize_entity_name(decomposed)
