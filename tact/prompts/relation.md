A reader was asked the question below about the document. The facts it needs
are all in the claims listed, but the question asks how they are related --
why one happened, what one resulted in, what one was for, how one works --
and no claim states that relation.

Question: {question}

Passage(s) that answer it:
{passages}

Claims (id: entity — relation: value):
{claims}

Write the relation claims the document states between these claims. Each one:

- "id": "c1", "c2", ... in order.
- "entity": the entity of the claim the relation starts from.
- "relation": the link, in snake_case ("results_in", "caused_by",
  "in_order_to", "mechanism_of", "led_to").
- "value_ref": the id, from the list above, of the claim the relation points to.
- "verbatim": a list holding the passage(s) that state the link -- the
  connective and its immediate context -- COPIED CHARACTER FOR CHARACTER from
  the document, as more than one string if the two halves are not adjacent.

Only relations the document actually states. If it states none, reply with an
empty list.

--- DOCUMENT ---
{document}
--- END DOCUMENT ---

Reply with JSON only, no prose and no code fence:
{{"claims": [{{"id": "c1", "entity": "...", "relation": "results_in", "value_ref": "...", "verbatim": ["..."]}}]}}
