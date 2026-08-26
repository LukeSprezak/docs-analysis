"""How a chunk points back at the document it came from — the two rules, written down once.

A chunk carries two names for its parent: `doc_id`, the namespaced identifier the system
assigns (`"{owner_id}::{filename}"` — see `UploadDocumentUseCase`), and `filename`, the name
the user gave the file. Which of the two comes first depends on who is asking, and that is
the part worth stating explicitly rather than leaving to be inferred from every call site:

* **Machines take `doc_id` first** (`parent_document_id`). Retrieval hands these ids to
  deletion and re-upload, where hitting another user's document would be a real failure, so
  the namespaced id wins over a name several users can share.
* **People take `filename` first** (`citation_label`). It is what a citation shows the user
  and what a golden set is written against; `doc_id` there would put an owner prefix in front
  of every source and match no reference answer.

Both fall back to the other name, so a chunk missing either key still identifies as something.
"""

from collections.abc import Mapping
from typing import Any

from .models import Document

# What a chunk with neither name identifies as. Deliberately not an exception: retrieval
# returning nothing at all is worse than returning a hit that cannot be traced back.
UNKNOWN_DOCUMENT_ID = "unknown"


def parent_document_id(metadata: Mapping[str, Any]) -> str:
    """The parent document's id, as retrieval reports it. Takes chunk metadata, not a Document
    — the adapters call this while building one."""
    return str(metadata.get("doc_id") or metadata.get("filename") or UNKNOWN_DOCUMENT_ID)


def citation_label(document: Document) -> str:
    """The parent document's human-readable name, for citations and golden-set matching."""
    return str(
        document.metadata.get("filename") or document.metadata.get("doc_id") or document.id
    )
