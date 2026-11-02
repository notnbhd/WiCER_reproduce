Extract every assertion in the document below that could ground a factual
question about it. Be exhaustive, not selective: a fact you leave out can never
be recovered later. Dates, figures, names, places, quantities, reasons, outcomes
and who-said-what all count.

Each claim is one assertion about one entity:

- "id": "c1", "c2", ... in order.
- "entity": one name from the entity list below, spelled exactly as listed.
- "relation": a short snake_case predicate ("opened_in", "employs", "founded_by").
- "value": the object of the assertion, as short as possible, with every
  number, name and unit copied exactly.
- "time": the date or period the assertion is about, as written in the
  document, or null.
- "speaker": the person or body the document attributes the assertion to
  ("according to X", a quotation, "X said"), or null.
- "verbatim": a list holding the shortest passage of the document that states
  the claim, COPIED CHARACTER FOR CHARACTER. Do not paraphrase, shorten words,
  fix typos or change punctuation: the passage is checked against the document
  and a claim whose passage is not found is discarded.

Relations between assertions are claims too. When the document links two of
your claims -- one causes, results in, enables, explains, or is the purpose of
the other -- add a claim with "relation" naming the link ("results_in",
"caused_by", "in_order_to", "mechanism_of"), "entity" the entity of the first
claim, "value_ref" set to the id of the second claim instead of "value", and
"verbatim" quoting the connective and its immediate context. If the two halves
of the link are not in one contiguous passage, give each piece as its own
string in the "verbatim" list, in document order.

Entities:
{entities}

--- DOCUMENT ---
{document}
--- END DOCUMENT ---

Reply with JSON only, no prose and no code fence:
{{"claims": [{{"id": "c1", "entity": "...", "relation": "...", "value": "...", "time": null, "speaker": null, "verbatim": ["..."]}}]}}
