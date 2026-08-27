"""Document identity: how an id is built, taken apart, and which name a caller should read.

Two different joins used to share the `::` marker, so a change to one read like a change to
the other. They are separate here, and this module is the only place either is written:

* **`::` namespaces a document under its owner** — `"{owner_id}::{filename}"`. Two users can
  upload a file with the same name without colliding on `documents.id`.
* **`#` joins a chunk to its parent document** — `"{document_id}#{chunk_index}"`. Chunk ids
  are never taken apart again; they only have to be unique.

A chunk carries two names for its parent: `doc_id`, the namespaced identifier above, and
`filename`, the name the user gave the file. Which of the two comes first depends on who is
asking, and that is the part worth stating explicitly rather than leaving to be inferred from
every call site:

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

OWNER_SEPARATOR = "::"
CHUNK_SEPARATOR = "#"


def namespaced_document_id(owner_id: str, doc_id: str) -> str:
    """The id a document is stored under — the owner's namespace in front of the file name."""
    return f"{owner_id}{OWNER_SEPARATOR}{doc_id}"


def strip_owner_namespace(document_id: str, owner_id: str) -> str:
    """The file name back out of a namespaced id, for citations and golden-set matching."""
    return document_id.removeprefix(f"{owner_id}{OWNER_SEPARATOR}")


def chunk_id(document_id: str, chunk_index: int) -> str:
    """The id of one chunk of a document."""
    return f"{document_id}{CHUNK_SEPARATOR}{chunk_index}"


def parent_document_id(metadata: Mapping[str, Any]) -> str:
    """The parent document's id, as retrieval reports it. Takes chunk metadata, not a Document
    — the adapters call this while building one."""
    return str(metadata.get("doc_id") or metadata.get("filename") or UNKNOWN_DOCUMENT_ID)


def citation_label(document: Document) -> str:
    """The parent document's human-readable name, for citations and golden-set matching."""
    return str(
        document.metadata.get("filename") or document.metadata.get("doc_id") or document.id
    )
