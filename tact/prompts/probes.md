Write {m} questions that test whether someone has read the document below.

- Spread them over the WHOLE document: divide it into {m} consecutive parts,
  beginning to end, and ask one question from each part.
- Each question must have one specific answer stated in the document: a
  figure, a date, a name, a place, who said something, why something
  happened, what it led to, or how it works. No opinion questions, and no
  questions the document does not answer.
- Do not quote the passage in the question; ask it the way a reader would.

For each question give:

- "question": the question.
- "answer": the correct answer, short but complete.
- "verbatim": a list holding the passage(s) of the document that state the
  answer, COPIED CHARACTER FOR CHARACTER. Do not paraphrase, shorten or fix
  anything: the passage is checked against the document and a question whose
  passage is not found is discarded. Use more than one passage only when the
  answer joins facts stated in different places.
- "kind": one of "date", "quantity", "person", "place", "attribution",
  "reason", "process", "other".
- "relational": true when the answer is a relation between facts -- why
  something happened, what it resulted in, what it was for, how it works --
  and false when it is a single fact.

--- DOCUMENT ---
{document}
--- END DOCUMENT ---

Reply with JSON only, no prose and no code fence:
{{"probes": [{{"question": "...", "answer": "...", "verbatim": ["..."], "kind": "date", "relational": false}}]}}
