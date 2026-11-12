# LLM abstention / unanswerable-question behavior under partial, compressed, fragmented or noisy context

Project framing: a reader LLM answers RepLiQA questions (about 20% unanswerable) from a compressed context of selected verbatim sentences. Sentence selection beats a contiguous-prefix truncation on answerable questions but answers more often when it should say "not mentioned". A claim-triple layout made the reader over-abstain on answerable questions.

Verification note: entries marked [V] had their arXiv abstract page or proceedings page fetched or returned by search this session (title, authors, ID confirmed). Entries marked [K] are well-known papers cited from prior knowledge; their arXiv IDs were not re-fetched this session, so check them before citing externally. Venue years follow the arXiv/proceedings pages where given.

## 1. Benchmarks and studies on abstention in RAG and long context

### Takeaway
Across 2024 to 2026 benchmarks, readers given retrieved or long context lean toward answering when the evidence is insufficient. Larger and reasoning models do not fix this, and AbstentionBench finds that reasoning models abstain less. The opposite failure, over-refusal, also shows up reliably when the context is irrelevant or when refusal is trained or prompted crudely. Any evaluation has to measure both directions.

### Cited Findings (annotated bibliography)
- [V] **AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions**: Kirichenko et al., arXiv June 2025 (NeurIPS 2025 per search results). https://arxiv.org/abs/2506.09038
  - Finding: covers 20 datasets and 20 frontier LLMs. Abstention is a weak spot. Reasoning models (DeepSeek R1 distill, s1.1) abstain about 24% less on average than their non-reasoning counterparts and often hallucinate missing context — [arXiv](https://arxiv.org/html/2506.09038v1)
  - Relevance: this tests whether using a reasoning or "thinking" reader would make our unanswerable regression worse. It also gives an LLM-judge abstention-scoring protocol we could reuse.
- [V] **Know Your Limits: A Survey of Abstention in Large Language Models**: Wen et al., TACL 2025 (arXiv v3 Feb 2025; Anthology entry 2025.tacl-1.26). https://arxiv.org/abs/2407.18418
  - Finding: the survey frames abstention through query, model and human-values lenses. It treats abstention as a spectrum from a full "I don't know" to hedging, and it reviews methods and metrics — [arXiv](https://arxiv.org/html/2407.18418v3)
  - Relevance: use it as the taxonomy and metric reference (abstention precision/recall, over-abstention) when reporting our two failure modes.
- [V] **Sufficient Context: A New Lens on Retrieval Augmented Generation Systems**: Joren et al. (Google), arXiv Nov 2024, v3 Apr 2025. The arXiv page lists no venue; the user and other sources identify it as ICLR 2025, which was not confirmed this session. https://arxiv.org/abs/2411.06037
  - Finding: large models (Gemini 1.5 Pro, GPT-4o, Claude 3.5) often answer wrongly instead of abstaining when the context is insufficient. Smaller models hallucinate or abstain even when the context is sufficient. Selective generation guided by a sufficient-context autorater raises the share of correct answers among answered questions by 2 to 10% — [arXiv abs](https://arxiv.org/abs/2411.06037). Per the search snippet, adding context paradoxically reduces abstention — [arXiv html](https://arxiv.org/html/2411.06037v3)
  - Relevance: this closely matches our case. Selected sentences that are topically relevant but insufficient are exactly what makes strong readers answer instead of abstain. The sufficiency autorater plus selective generation is a ready mitigation.
- [V] **RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models**: Muhamed et al., arXiv Oct 2025. https://arxiv.org/abs/2510.10390
  - Finding: covers 30+ models, 176 perturbation strategies, 6 uncertainty categories and 3 intensities. Refusal accuracy falls below 50% on multi-document tasks. The failures are both overconfidence and overcaution. Neither scale nor extended reasoning helps. Detection and categorization are separable skills, and selective refusal is trainable — [arXiv](https://arxiv.org/abs/2510.10390)
  - Relevance: it documents both of our failure modes (sentence selection leads to overconfidence, the triple layout to overcaution) as distinct, perturbation-sensitive behaviors. Its perturbation generator could stress-test our compressed contexts.
- [V] **Unanswerability Evaluation for Retrieval Augmented Generation (UAEval4RAG)**: Peng et al., arXiv Dec 2024, v3 Apr 2025, no venue listed. https://arxiv.org/abs/2412.12300
  - Finding: proposes a six-category taxonomy of unanswerable queries with metrics for unanswered ratio and acceptable ratio. Retriever, reranker, LLM and prompt choices strongly trade off answerable accuracy against correct rejection — [arXiv](https://arxiv.org/abs/2412.12300)
  - Relevance: confirms that a pipeline component, here our context builder, can move the answerable/unanswerable tradeoff by itself, so both sides need reporting.
- [V] **Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse (Trust-Score / Trust-Align)**: Song et al., ICLR 2025 Oral. https://arxiv.org/abs/2409.11242
  - Finding: ICL/prompting does not adapt LLMs well to grounded refusal. Trust-Align alignment substantially improves correct refusal and citation quality — [arXiv](https://arxiv.org/abs/2409.11242)
  - Relevance: shows that prompt-only fixes have a ceiling, which matters if our abstention-prompt tweaks plateau.
- [V] **FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"**: Ming et al., ICLR 2025. https://arxiv.org/abs/2410.03727
  - Finding: 4.9K questions across unanswerable, inconsistent and counterfactual contexts; SOTA models often fail to stay faithful, and size does not reliably help — [arXiv](https://arxiv.org/abs/2410.03727), [FaithfulRAG citing it](https://arxiv.org/pdf/2506.08938). A third-party review reports one model scoring 76.8% on original contexts and 7.4% on the unanswerable variants — [Liner review](https://liner.com/review/faitheval-can-your-language-model-stay-faithful-to-context-even) (secondary source, not checked against the paper).
  - Relevance: its unanswerable subset removes the evidence while keeping a coherent context, which is close to what our sentence selection does to unanswerable RepLiQA items.
- [V] **Towards Reliable Long-Context Reasoning: Detecting Unanswerable Questions via FactGuard**: arXiv Apr 2025. https://arxiv.org/abs/2504.05607
  - Finding: 25,220 examples with 4K to 128K contexts, made unanswerable by evidence removal and entity substitution. LLMs hallucinate overconfidently as the context grows. Fine-tuning raised an 8B model's unanswerable recall from 0.09 to 0.62 — [arXiv html](https://arxiv.org/html/2504.05607) (figures taken from the search snippet)
  - Relevance: its evidence-removal construction mirrors compression that drops the answer sentence. It suggests that fine-tuning, not prompting, is what moves unanswerable recall.
- [V] **Do Retrieval Augmented Language Models Know When They Don't Know?**: Zhou et al., AAAI 2026 (pp. 35158–35166). https://arxiv.org/abs/2509.01476
  - Finding: with all-irrelevant retrieval, RALMs refuse questions they could answer. A third-party summary puts over-refusal above 55%, which was not verified in the paper. In-context fine-tuning reduces over-refusal while R-Tuning increases it. Fewer refusals do not imply better calibration. Experiments use about 7B models — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40822), [papernotes summary](https://en.papernotes.org/AAAI2026/information_retrieval/do_retrieval_augmented_language_models_know_when_they_dont_know/)
  - Relevance: this is the over-abstention side. It is a plausible mechanism for our claim-triple layout: when context looks off-distribution or weakly relevant, the reader refuses.
- [V] **LIT-RAGBench: Benchmarking Generator Capabilities of Large Language Models in Retrieval-Augmented Generation**: arXiv Mar 2026. https://arxiv.org/abs/2603.06198
  - Finding (search snippet only): includes an Insufficient Evidence category and reports a tendency toward over-abstention even when a valid answer is possible — [arXiv](https://arxiv.org/pdf/2603.06198)
  - Relevance: recent evidence that over-abstention is a common generator failure, not something specific to our setup.
- [V] **Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG (RINSE)**: Chen, Hua, Xiao, arXiv Sep 2026. https://arxiv.org/abs/2609.37469
  - Finding: 12 generators instructed to abstain still answered 40.0% to 99.3% of insufficient-evidence questions. The pre-generation sufficiency ranker RINSE scores 0.837 (chance 0.5), against 0.746 for the best of 10 prior methods and 0.784 for a frontier API model, at 36.5 ms per question — [arXiv](https://arxiv.org/abs/2609.37469)
  - Relevance: the title states our problem exactly. Selected sentences are relevant without being sufficient. A cheap pre-reader sufficiency check is a direct candidate mitigation.
- [V] **Prompt-Based Abstention Fails Under Misleading Context: A Controlled Study of Small Frozen RAG Models**: Setiawan, arXiv Aug 2026 (committed to AACL-IJCNLP 2026). https://arxiv.org/abs/2608.22228
  - Finding: covers 3.8B to 8B frozen models on NQ and HotpotQA. Models abstain reliably when evidence is absent but answer 41.6% of misleading-context questions despite abstention prompts, and 63% of those wrong answers echo the planted entity. A generator-side conflict check cuts wrong answers to 13.3% but drops correct answers. CoT adds little — [arXiv](https://arxiv.org/abs/2608.22228)
  - Relevance: an important distinction for us. Abstention holds when the context is clearly off-topic and fails when the context is near-miss or plausible, which is what selected sentences from the right document provide.
- [V] **Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework**: Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.133.pdf
  - Finding (search snippet): RAG models given noisy retrieval give confident wrong answers, and calibration-based thresholding can remain overconfident — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.133.pdf)
  - Relevance: background for treating abstention as risk control. Full title and authors were not confirmed from the page this session; only the Anthology ID was.

### Inferences
- Our unanswerable regression fits a reported pattern. Context that is topically relevant but insufficient raises false answering more than context that is absent or irrelevant (Sufficient Context, RINSE, Setiawan 2026). Sentence selection is designed to maximize topical relevance, so on unanswerable questions it produces exactly this near-miss context. Truncation's contiguous prefix more often contains nothing on-topic, so the reader abstains.
- Our claim-triple over-abstention fits the over-refusal literature (Zhou et al. AAAI 2026, RefusalBench, LIT-RAGBench). A format that looks less like answer-bearing prose probably lowers the reader's perceived sufficiency. This is an inference; no paper tests triple layouts specifically.

### Gaps
- No paper found that evaluates RepLiQA specifically for abstention under compression.
- Exact per-model abstention numbers for AbstentionBench and Sufficient Context were not extracted; the abstract pages give only aggregates.

## 2. Does compression or retrieval noise change hallucination and abstention rates? Does fragmentation cause false confidence?

### Takeaway
Direct evidence is thin but growing. One 2026 paper shows that compression which deletes answer evidence causes hallucination rather than abstention, and that a small trained adapter fixes most of it. The best-supported mechanistic result (AbsenceBench) is that LLMs are very poor at noticing what has been removed from a text, and explicit omission markers help a lot. No study was found that isolates sentence-level fragmentation or loss of coherence as a cause of false confidence.

### Cited Findings
- [V] **Compression-Aware Abstention: Teaching LLMs to Refuse When KV-Compression Masks Remove Answer Evidence**: Khodabandehlou & Krishnamachari, arXiv Aug 2026, EMNLP 2026 GroundLM workshop. https://arxiv.org/abs/2608.29934
  - Finding: when compression removes the answer evidence, base models hallucinate. A 10.1M-parameter LoRA trained on about 2.6K MuSiQue examples cuts hallucinations by 97% under prompt-style truncation and keeps correct answering. Prompt-only abstention baselines over-abstain on answerable high-retention examples — [arXiv](https://arxiv.org/abs/2608.29934)
  - Relevance: this is the closest paper to our setting. It shows both of our failure modes: compression without evidence leads to hallucination, and prompt-only abstention leads to over-refusal. It also suggests a learned conditional policy is the fix.
- [V] **AbsenceBench: Language Models Can't Tell What's Missing**: Fu et al., NeurIPS 2025 Datasets & Benchmarks (published as "Absence Bench: Language Models Can't See What's Missing"). https://arxiv.org/abs/2506.11440
  - Finding: models are asked to find deliberately removed content given both the original and the edited text. Claude-3.7-Sonnet reaches only about 69.6% F1 at about 5K tokens. The authors attribute this to attention: gaps have no tokens to attend to. Inserting explicit placeholders at the omissions improves performance by 35.7% on average — [arXiv html](https://arxiv.org/html/2506.11440v1), [NeurIPS](https://proceedings.nips.cc/paper_files/paper/2025/hash/36b31e1bb8ecd4f4081686448e9eff2d-Abstract-Datasets_and_Benchmarks_Track.html)
  - Relevance: this is the strongest evidence for a fragmentation mechanism. A sentence-selected context hides its gaps, so the reader cannot tell that the dropped sentences might have held the answer. It directly supports testing "[...]" omission markers between non-adjacent selected sentences.
- [V] **An Empirical Study on Prompt Compression for Large Language Models**: Zhang et al., ICLR 2025 Building Trust workshop. https://arxiv.org/abs/2505.00019
  - Finding: covers six compression methods. Compression hurts more in long contexts, and moderate compression can improve LongBench results. The paper analyzes hallucination with information-loss and altered-semantics types, plus word omission, but the abstract does not quantify these — [arXiv](https://arxiv.org/abs/2505.00019)
  - Relevance: shows that compression can help accuracy, which matches our answerable gain. Its hallucination section should be read for whether information loss produces fabricated answers rather than abstention.
- [V] **How Much Do LLMs Hallucinate in Document Q&A Scenarios? A 172-Billion-Token Study Across Temperatures, Context Lengths, and Hardware Platforms**: arXiv Mar 2026. https://arxiv.org/abs/2603.08274
  - Finding (search snippet): the hallucination floor is non-zero even under optimal conditions and rises steeply with context length — [arXiv](https://arxiv.org/html/2603.08274v1)
  - Relevance: background. Shorter compressed contexts should lower length-driven hallucination, so our unanswerable regression likely comes from content choice, not length.
- [V] Sufficient Context (above): adding context reduced appropriate abstention — [arXiv html](https://arxiv.org/html/2411.06037v3)
- [V] Setiawan 2026 (above): absent evidence leads to reliable abstention, while misleading or near-miss evidence leads to answering despite instructions — [arXiv](https://arxiv.org/abs/2608.22228)
- [K] **The Power of Noise: Redefining Retrieval for RAG Systems**: Cuconasu et al., SIGIR 2024. https://arxiv.org/abs/2401.14887
  - Finding: related-but-non-answering ("distracting") passages hurt accuracy more than random passages, and random noise can even help.
  - Relevance: topically similar distractors are the harmful kind, and sentence selection by similarity produces exactly those on unanswerable questions.

### Inferences
- A plausible two-part mechanism for our regression combines the cited findings. (a) Selection removes the visible "gap" signals, which AbsenceBench shows models cannot detect without markers. (b) Selection keeps high-similarity near-miss sentences, which per Sufficient Context, Power of Noise and Setiawan raise false answering. Truncation keeps a coherent prefix in which absence is more legible.
- The workshop paper's finding that "prompt-only abstention over-refuses at high retention" predicts that simply strengthening our "say not mentioned" instruction will trade the unanswerable gain for answerable loss.

### Gaps
- No study was found that directly compares contiguous truncation with extractive sentence selection on unanswerable-question behavior. That would be a novel contribution.
- No paper was found that tests sentence-order or coherence perturbation (shuffling, de-contextualized sentences) against QA abstention rates. A search returned only adjacent work, such as the translation coherence study at https://arxiv.org/pdf/2305.03573 (2023, out of range).

## 3. Mitigations: omission signaling, sufficiency classifiers, calibrated abstention prompts, selective generation

### Takeaway
The approaches with measured gains are: (1) explicit omission markers (+35.7% omission detection, AbsenceBench); (2) a separate sufficiency judgment before or alongside generation, used for selective generation (+2 to 10% precision for Sufficient Context; 0.837 ranking for RINSE); (3) small trained adapters or alignment instead of prompts (Compression-Aware Abstention, Trust-Align, FactGuard). Prompt-only abstention is repeatedly shown to be weak or to cause over-refusal.

### Cited Findings
- Omission markers: explicit placeholders at removed spans improve missing-content detection by 35.7% on average — [AbsenceBench](https://arxiv.org/html/2506.11440v1). Not tested on QA abstention specifically.
- Sufficiency autorater plus selective generation: +2 to 10% correct-among-answered for Gemini, GPT and Gemma — [Sufficient Context](https://arxiv.org/abs/2411.06037)
- Pre-generation sufficiency ranker: 0.837 vs 0.746 for the best prior method, 36.5 ms per query — [RINSE](https://arxiv.org/abs/2609.37469)
- Learned compression-aware abstention: −97% hallucination under truncation, and correct answers preserved — [Khodabandehlou & Krishnamachari 2026](https://arxiv.org/abs/2608.29934)
- Prompting has limits: ICL does not adapt models well to grounded refusal — [Trust-Align](https://arxiv.org/abs/2409.11242). Abstention prompts still leave 41.6% answered on misleading context, and CoT adds little — [Setiawan 2026](https://arxiv.org/abs/2608.22228). Instructed generators answer 40 to 99% of insufficient cases — [RINSE](https://arxiv.org/abs/2609.37469)
- Training choice matters for over-refusal: in-context fine-tuning reduces it, R-Tuning increases it — [Zhou et al. AAAI 2026](https://arxiv.org/abs/2509.01476)
- Generator-side conflict checks cut wrong answers (13.3%) but cost coverage; an NLI verifier restores coverage but fails when parametric memory agrees with the wrong context — [Setiawan 2026](https://arxiv.org/abs/2608.22228)
- Selective refusal is trainable, and detection and categorization are separable — [RefusalBench](https://arxiv.org/abs/2510.10390)

### Inferences
- The cheapest experiment for us is inserting "[...]" between non-adjacent selected sentences, plus an "these are excerpts; omitted text may contain the answer" framing. AbsenceBench gives the mechanistic rationale. The risk is over-abstention (see Section 1); the measured gain is on omission detection, not on QA.
- A second cheap option is a separate sufficiency call (same model or small classifier) on the compressed context, gating the answer. This is a known method with measured precision gains, but it costs answerable coverage.

### Gaps
- No paper was found that measures "excerpt framing" prompts for QA abstention.
- Conformal abstention and selective generation papers from 2024 (for example Yadkori et al., "Mitigating LLM Hallucinations via Conformal Abstention", believed to be arXiv 2405.01563) were not verified this session.

## 4. Reader sensitivity to coherence, position ("lost in the middle"), and distractors

### Takeaway
The position and distractor effects are well established: middle-position evidence is under-used, and topically related distractors are the most harmful. Coherence as an isolated variable is under-studied. Adjacent safety work shows that reasoning over fragments spread through a context changes model refusal behavior.

### Cited Findings
- [K] **Lost in the Middle: How Language Models Use Long Contexts**: Liu et al., TACL 2024 (arXiv 2023). https://arxiv.org/abs/2307.03172
  - Finding: QA accuracy is U-shaped over answer position, lowest in the middle.
  - Relevance: a compressed context is short, so the position effect should be weaker than in long-context truncation. The ordering of selected sentences still matters.
- [K] **Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization**: Hsieh et al., Findings ACL 2024. https://arxiv.org/abs/2406.16008
  - Finding: the U-shaped bias stems from positional attention bias, and calibrating it improves middle-context use.
  - Relevance: a mechanistic counterpart to the position effect.
- [K] Power of Noise (Section 2): related distractors are more harmful than random ones — https://arxiv.org/abs/2401.14887
- [V] **Do Reasoning LLMs Refuse What They Infer in Long Contexts?**: Fu et al., arXiv Feb 2026 (v2 May 2026). https://arxiv.org/abs/2602.08874
  - Finding: 15 LLMs, contexts up to 64K tokens. Refusal (of harmful requests) drops sharply when the objective must be reconstructed from fragments scattered across the context rather than retrieved directly, and worse in longer contexts. Failures are not mainly retrieval errors — [arXiv](https://arxiv.org/abs/2602.08874)
  - Relevance: tangential, since it concerns safety refusal, not unanswerability. It is still evidence that fragmented evidence changes refusal decisions in a way direct evidence does not.
- [V] RefusalBench: performance drops on multi-document (more fragmented) inputs below 50% — https://arxiv.org/abs/2510.10390

### Inferences
- Sentence selection produces multi-source-like, de-contextualized input. RefusalBench's multi-document drop and the fragment-reconstruction result suggest fragmentation by itself degrades refusal calibration. This is suggestive, not direct, evidence.

### Gaps
- No 2025–2026 paper was found that isolates discourse coherence (shuffled or decontextualized sentences vs contiguous text) as a factor in abstention. Searches for "context coherence shuffled sentences hallucination" returned nothing on target.
- Lost in the Middle, Found in the Middle and Power of Noise are cited from prior knowledge [K]; their IDs were not re-fetched this session.
