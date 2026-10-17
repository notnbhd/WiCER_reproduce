# Compiler and diagnosis prompts

**These prompts are not from the paper.** arXiv:2605.07068v1 publishes exactly
one prompt: the LLM-as-judge rubric (Appendix B), which lives verbatim in
`wicer_core/judge.py`. The compilation and diagnosis prompts are described in
prose (Sections 6.1 and 7.1) but never given.

They are reconstructed here from the constraints the paper states:

* the compiler acts as a knowledge engineer and is **blind to the evaluation
  questions** (Section 6.1);
* it targets a compression level of ~75 / ~50 / ~25 % of the source
  (Section 6.1) -- and characteristically overshoots, which is the finding, so
  nothing here tries to enforce the target (Section 6.3, spec.md Section 5);
* preserved facts are a **hard constraint**: included verbatim, and placed
  before any budget is spent on general coverage (Section 7.1, Appendix I);
* diagnosis extracts ~50-100 words per failure against ~700 words per source
  document (Section 7.2).

This is the single largest source of divergence from the original. See
`DEVIATIONS.md`.
