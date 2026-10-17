A downstream assistant was asked a question about the source document below.
It was working from a compressed wiki rather than the document itself, and it
answered wrongly -- the information it needed had been dropped in compilation.

Your job is to extract the specific facts that answer had to contain.

Question: {question}

Correct answer: {gold}

What the assistant said instead: {generated}

--- SOURCE DOCUMENT ({document_id}) ---
{document}
--- END SOURCE DOCUMENT ---

Extract between {min_words} and {max_words} words of the specific facts from
this document that the correct answer depends on. Include the surrounding
detail needed to make them usable -- the entities they attach to, their dates,
their figures -- but nothing beyond that.

Do NOT summarise the document. Do NOT restate the question or the answer. Do
NOT explain your reasoning. Emit only the extracted facts, as plain prose or a
short list.
