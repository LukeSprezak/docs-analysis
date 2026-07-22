"""Entity identity: which two mentions are the same thing.

An LLM names the same entity differently every few passages — `"Postgres"`, `"postgres"`,
`" Postgres "`, `"Postgres."`. Each spelling used to become its own node, so facts about one
thing scattered across several of them and traversals reached only a fraction of what the
corpus actually said. Merging on a normalized key instead of the raw name collapses those.

**What this deliberately does not do.** It is a deterministic string rule, not a resolver:

* `"Postgres"` and `"PostgreSQL"` stay separate. Catching those needs an alias list or fuzzy
  matching, and fuzzy matching on entity names is dangerous — `"Python 2"` and `"Python 3"`
  are one edit apart and must never merge. A conservative rule that under-merges is the safe
  default; over-merging silently fuses unrelated facts and is far harder to notice.
* Diacritics are preserved. Folding them would merge `"Łódź"`/`"Lodz"`, but it would also
  merge genuinely different Polish words, and this corpus is Polish-first.

The normalized form is the merge key only. The first spelling encountered stays as the
display `name` and is what gets cited, so normalization never leaks into the output.
"""

import re
import unicodedata

# Quotes, brackets and sentence punctuation that cling to the edges of an extracted name.
_EDGE_PUNCTUATION = re.compile(r"^[\s\"'“”„«»(\[{.,;:!?-]+|[\s\"'“”„«»)\]}.,;:!?-]+$")
_INNER_WHITESPACE = re.compile(r"\s+")


def normalize_entity_name(name: str) -> str:
    """The key two mentions must share to be treated as the same entity.

    Composes accents to a canonical form (NFC), drops edge punctuation, collapses runs of
    whitespace and casefolds. Falls back to the plain casefolded input when stripping would
    leave nothing — a name made entirely of punctuation is odd, but it must not collapse into
    the empty key and merge with every other such name.
    """
    cleaned = unicodedata.normalize("NFC", name)
    cleaned = _EDGE_PUNCTUATION.sub("", cleaned)
    cleaned = _INNER_WHITESPACE.sub(" ", cleaned).strip().casefold()
    return cleaned or name.strip().casefold()
