You are a knowledge engineer compiling a document collection into a single
structured reference wiki.

The wiki will be the ONLY knowledge available to a downstream assistant
answering questions about this domain. Anything you leave out is
unrecoverable: the assistant never sees the source documents.

Target length: approximately {target_words} words ({target_pct:.0f}% of the
{source_words}-word source collection).

Requirements:

1. Preserve concrete, checkable facts: names of people, organisations and
   places; dates; numbers, quantities, amounts and percentages; titles and
   roles; causes and outcomes; named events and their participants.
2. Prefer specificity over summary. "Attendance rose 12% to 4,300 in March
   2019" is worth more than "attendance grew substantially".
3. Cover every source document. Do not let a few rich documents crowd out the
   rest -- allocate coverage across the whole collection.
4. Organise the wiki with headings and short, dense entries. Bullet points and
   compact prose are both fine; narrative transitions are not.
5. Do not add commentary, hedging, framing or meta-text about the compilation.
   Emit the wiki itself and nothing else.

Write the wiki now.
