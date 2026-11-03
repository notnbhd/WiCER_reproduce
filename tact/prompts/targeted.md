A reader was asked the question below about the document, and the claims
already extracted from the document do not cover the passage that answers it.
Extract the claims that passage supports.

Question: {question}

Passage(s) that answer it:
{passages}

Use the same claim format as the rest of the ledger:

- "id": "c1", "c2", ... in order.
- "entity": preferably one of the entities below, spelled exactly as listed;
  a new entity only if none of them fits.
- "relation": a short snake_case predicate.
- "value": the object of the assertion, every number, name and unit copied
  exactly.
- "time": the date or period the assertion is about, as written, or null.
- "speaker": who the document attributes it to, or null.
- "verbatim": a list holding the shortest passage stating the claim, COPIED
  CHARACTER FOR CHARACTER from the document. A claim whose passage is not
  found in the document is discarded.

Extract every assertion in the passage(s) that the question could need, not
only the one that answers it.

Entities already in the ledger for this document:
{entities}

--- DOCUMENT ---
{document}
--- END DOCUMENT ---

Reply with JSON only, no prose and no code fence:
{{"claims": [{{"id": "c1", "entity": "...", "relation": "...", "value": "...", "time": null, "speaker": null, "verbatim": ["..."]}}]}}
