"""Lucene query escaping, shared by both Neo4j adapters.

Neo4j's full-text indexes are Lucene indexes, so every adapter that reaches them through
`db.index.fulltext.queryNodes` passes user input through the same neutralization. One copy,
because this is a correctness/security control: a character added to the list in one adapter
and missing in the other leaves one of the two candidate sources unprotected — which shows up
as skewed results or a syntax error mid-request for half the retrieval, not as a clean break.
"""

# Lucene's operators. A user's question would otherwise have `:` read as a field separator,
# `~` as a fuzzy match, and so on.
LUCENE_SPECIAL_CHARACTERS = r'+-&|!(){}[]^"~*?:\/'


def escape_lucene(query: str) -> str:
    """Escapes Lucene syntax in user input so it is matched as text, not parsed as operators."""
    return "".join(
        f"\\{character}" if character in LUCENE_SPECIAL_CHARACTERS else character
        for character in query
    )
