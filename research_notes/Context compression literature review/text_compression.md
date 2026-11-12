# Query-agnostic (task-agnostic) text context compression for LLM readers, 2024–2026

Project framing (for relevance lines): we compile a corpus into a fixed budget (~45% of source, i.e. ~2.2x) before questions are known; reader answers RepLiQA-style QA. Our findings: (F1) verbatim source sentences beat paraphrased/structured claims; (F2) coherent contiguous text beats max-coverage selection; (F3) dropping connective text breaks anaphora and makes the reader abstain.

Verification note: entries marked [verified] had their arXiv/venue page or a direct index hit checked in this session. Entries marked [prior knowledge] are well-known foundational papers whose IDs I am confident of, but whose pages I did not re-open this session; the numeric findings on those are from memory of the abstracts and should be spot-checked before quoting.

## Q1. LLMLingua family, Selective Context, RECOMP and 2025–2026 successors: what do they find about query-agnostic vs query-aware compression?

### Takeaway
Every study that compares the two finds query-aware compression beats query-agnostic at the same ratio, and theory (rate-distortion) says the gap is fundamental, not just an engineering gap. The query-agnostic methods that hold up best keep source tokens/sentences verbatim (LLMLingua-2, sentence-level TPC/CPC, KVzip at the KV level). The newest 2026 benchmark (GACC-Bench) shows question-blind token pruning collapses at aggressive budgets.

### Cited Findings
Foundational (brief):
- **Selective Context**: "Compressing Context to Enhance Inference Efficiency of Large Language Models", Yucheng Li et al., EMNLP 2023, https://arxiv.org/abs/2310.06201 [prior knowledge]. Finding: query-agnostic pruning of low self-information lexical units (tokens/phrases/sentences) at ~50% reduction with modest drops on summarization/QA/conversation. Relevance: the closest classic analogue to our ~45% question-blind budget. It works at the lexical-unit level, so it produces the fragmented text we found harmful (F3).
- **LLMLingua**: "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", Huiqiang Jiang et al., EMNLP 2023, https://arxiv.org/abs/2310.05736 [prior knowledge]. Finding: coarse-to-fine perplexity-based token dropping with a budget controller, up to ~20x on reasoning/ICL prompts. Relevance: the token-pruning baseline; output is ungrammatical text.
- **LongLLMLingua**: "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression", Huiqiang Jiang et al., ACL 2024, https://arxiv.org/abs/2310.06839 [prior knowledge]. Finding: adds question-aware coarse-to-fine compression and document reordering; reports gains on NaturalQuestions with ~4x fewer tokens (abstract quotes up to +21.4%). Relevance: its gains depend on knowing the question, which we don't at compile time.
- **LLMLingua-2**: "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression", Zhuoshi Pan et al., Findings of ACL 2024, https://arxiv.org/abs/2403.12968 [prior knowledge]. Finding: task-agnostic token classification (keep/drop) trained on GPT-4 extractive distillation data (MeetingBank). Because it only deletes, the output stays faithful to the source. Reports 2x–5x compression and 1.6x–2.9x end-to-end speedup, with out-of-domain generalization to LongBench/ZeroSCROLLS. Relevance: the main query-agnostic token-level baseline we should beat or match at ~2.2x; it agrees with F1 (it extracts rather than generates) but not with F3 (it fragments sentences).
- **RECOMP**: "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation", Fangyuan Xu, Weijia Shi, Eunsol Choi, ICLR 2024, https://arxiv.org/abs/2310.04408 [prior knowledge]. Finding: trains both an extractive (sentence selector) and an abstractive (summarizer) compressor, reaching roughly 6% of tokens with small loss. The abstractive compressor may output an empty string when docs are irrelevant ("selective augmentation"). Relevance: it is query-aware. Its extractive/abstractive split is the template for our verbatim-vs-paraphrase comparison (F1).

Theory and 2024–2026 work:
- **Nagle et al.**, "Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models", Adway Girish, Alliot Nagle et al., NeurIPS 2024, https://arxiv.org/abs/2407.15504 [verified]. Finding: it computes the optimal rate-distortion curve for hard-prompt compression as a linear program. Query-aware compression significantly beats query-agnostic, and there is a large gap between existing methods and the optimum. Only their query-aware, variable-rate LLMLingua-2 variant ("Adaptive QuerySelect") beats the *optimal* query-agnostic strategy. Relevance: this formally bounds what any question-blind compiler (ours) can achieve. Variable-rate allocation (per document or segment) is the lever left to us.
- **Jha et al.**, "Characterizing Prompt Compression Methods for Long Context Inference", Siddharth Jha et al. (UC Berkeley), arXiv July 2024, https://arxiv.org/abs/2407.08892 [verified; read full HTML]. Findings on LongBench with GPT-3.5/Mixtral/DBRX:
  - Query-agnostic abstractive (offline chunk summaries by Mistral-7B) trails extractive by 3–5 pts on summarization and 10–15 pts on MultiFieldQA at the same ratio.
  - Query-aware summarization is 3–6 pts better than query-agnostic (e.g. HotpotQA 52.23 vs 48.01).
  Relevance: the directly relevant number is that question-blind summaries lose 10–15 pts on single-doc QA versus extraction (F1).
- **Zhang et al.**, "An Empirical Study on Prompt Compression for Large Language Models", Zheng Zhang et al., ICLR 2025 Building Trust workshop, https://arxiv.org/abs/2505.00019 [verified]. Finding: it compares KiS, SCRL, Selective Context, LLMLingua, LongLLMLingua and LLMLingua-2 across 13 datasets.
  - Compression hurts more in long contexts, though moderate compression even improved LongBench.
  - Question-aware LongLLMLingua was best on long-context QA, and task-agnostic LLMLingua-2 was best on reconstruction and summarization.
  Relevance: query-aware wins on QA. "Moderate compression can help" matches our recovery at ~2.2x.
- **TPC**, "Task-agnostic Prompt Compression with Context-aware Sentence Embedding and Reward-guided Task Descriptor", Barys Liskavets et al., arXiv Feb 2025, https://arxiv.org/abs/2502.13374 [verified]. Finding: a sentence-level, question-free compressor. It generates a pseudo task description, scores sentences by context-aware embeddings against it, and claims SOTA on LongBench/ZeroSCROLLS in prompt-agnostic settings. The abstract gives no numbers; check the tables. Relevance: the closest published design to ours (question-blind, keeps whole verbatim sentences). The "generate a task descriptor first" trick could be adapted to "expected RepLiQA question types".
- **CPC**, "Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference", Barys Liskavets et al., AAAI 2025, https://arxiv.org/abs/2409.01227 [verified exists; venue prior knowledge]. Finding: argues that token-level removal yields "non-coherent and grammatically incorrect" sentences that cause performance drops. Sentence-level selection is up to 10.93x faster than the best token-level method. It is query-aware. Relevance: supports F3 (keep whole sentences).
- **GACC-Bench**, "GACC-Bench: Benchmarking Goal-Adaptive Context Compression Ability across Large Language Models", Weiting Liu et al. (Fudan/JD), arXiv Oct 2026, https://arxiv.org/abs/2610.21893 [verified; read]. Setup: 500 goals, 2,571 questions, 281 contexts; the compressor sees the context, a high-level goal and a budget, but not the questions. Findings:
  - At 5% retention, the best LLM compressors reach ~42–43% micro accuracy vs 99.4% with full context; at 10%, ~57%.
  - Own-goal conditioning beats no-goal in every model/budget, and a mismatched goal is worst.
  - At 5%, question-blind LLMLingua-2 scores 11.52 micro, and all three pruning baselines (LLMLingua, LLMLingua-2, LongLLMLingua) score 0 macro.
  Relevance: the closest 2026 benchmark to our "questions unknown" setting. Its budgets (5–20%) are far harsher than our 45%. The "goal" (vs no goal) gain suggests telling our compiler the QA style should help.
- **Belcak et al.**, "You don't need long context", Peter Belcak et al. (NVIDIA), arXiv Nov 2026, https://arxiv.org/abs/2611.05306 [verified abstract only]. Finding (abstract): frontier models lose little to no accuracy on long-horizon/long-input tasks when kept to short contexts via "frequent compaction". Relevance: tangential, since compaction there is agentic and iterative; it signals 2026 interest in compaction over long context.
- **KVzip**, "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction", Jang-Hyun Kim et al., NeurIPS 2025 (oral), https://arxiv.org/abs/2505.23416 [verified]. Finding: it scores KV pairs by how well the LLM can *reconstruct the original context* from them. This gives a question-blind 3–4x KV reduction and ~2x decoding speedup with negligible loss across QA/retrieval/reasoning/code, up to 170K tokens. Relevance: the KV-level analogue of our compiler. Its "reconstructability" objective is a principled query-agnostic importance score we could borrow for sentence selection.

### Inferences
- Across text-level studies, "query-agnostic + extractive + sentence-granular" is the best-performing corner that remains available without the question. Our F1/F3 sit squarely in it.
- The rate-distortion result implies our remaining headroom is in *variable-rate* allocation (spend budget unevenly across docs and sections), not in a better per-token scorer.

### Gaps
- I found no paper that evaluates query-agnostic compression specifically on RepLiQA, or at exactly ~2x with an unanswerable-question slice.
- TPC's numeric comparison against LLMLingua-2 was not visible on the abstract page.

## Q2. Extractive vs abstractive accuracy, and coherence/fluency effects (does token pruning or sentence fragmentation hurt QA?)

### Takeaway
The independent benchmarks consistently rank extractive (sentence/chunk) compression above both abstractive summarization and token pruning on QA. Token pruning's deficit is explicitly attributed to broken grammar and sentence comprehension. Compression of any kind raises hallucination, mainly through information loss, and ungrammatical compressed text is suspected of encouraging it.

### Cited Findings
- **Jha et al. 2024** (https://arxiv.org/abs/2407.08892):
  - Extractive reranker-based compression "often outperforms all the other approaches", up to 10x with minimal loss and sometimes gains (2WikiMQA +7.89 at 7.75x).
  - LongLLMLingua and reranker+LongLLMLingua were typically *worst*; hyperparameter sweeps did not help.
  - The authors attribute token pruning's "lackluster performance" to "the disruption of grammar and sentence comprehension caused by unstructured pruning". Token pruning is "better suited for aggregation-style tasks" (summarization) than QA.
  - On text-to-SQL, join accuracy fell 0.63 → 0.37 as pruning rose 1.62x → 4.29x.
  Relevance: the strongest independent support for F1 + F3.
- **Zhang et al. 2025** (https://arxiv.org/abs/2505.00019):
  - All six methods increased hallucination relative to the uncompressed prompt, and random selection was worst.
  - Information loss was the main trigger. LLMLingua-2 had the lowest hallucination on reconstruction/summarization, and LongLLMLingua on long-context QA.
  - The authors suggest compressed prompts can be ungrammatical or overly terse and may encourage hallucination. In an appendix example, Claude-3-Haiku itself calls a compressed prompt "incomplete and grammatically flawed".
  Relevance: a Haiku reader flagging fragmented input matches our observation that fragmented context makes the reader abstain (F3), though they report hallucination rather than abstention.
- **CPC** (https://arxiv.org/abs/2409.01227): an explicit argument that intra-sentence token removal makes text non-coherent and ungrammatical, hurting the LLM. It motivates whole-sentence selection. Relevance: F3. Caveat: authored by the method's proponents.
- **Łajewska et al.**, "Understanding and Improving Information Preservation in Prompt Compression for LLMs", Weronika Łajewska et al., Findings of EMNLP 2025, https://arxiv.org/abs/2503.19114 / https://aclanthology.org/2025.findings-emnlp.949 [verified]. Findings:
  - Soft and hard SOTA compressors "struggle to preserve key details", which limits performance on complex tasks.
  - Downstream accuracy alone is an insufficient metric; they add grounding (BERTScore to source) and entity preservation.
  - Controlling granularity in a soft-prompt method gave +23% downstream, +8 BERTScore grounding and 2.7x more entities preserved.
  Relevance: entity/detail-level loss matches our held-out finding that failures are detail-level. Their entity-preservation metric is a cheap diagnostic for our compiled wikis.
- **EXIT**, "EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation", Taeho Hwang et al. (KAIST), Findings of ACL 2025, https://arxiv.org/abs/2412.12559 [verified]. Finding: classifies each sentence while conditioning on its full document, then reassembles the kept sentences in original order. It beats both prior extractive (independent sentence scoring) and abstractive compressors on single/multi-hop QA, with latency of ~1s vs seconds. It frames abstractive compression as slow and hallucination-prone. Relevance: supports F2. Sentence decisions should see document context, and output order should be preserved. It is query-aware.
- **Provence**, "Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation", Nadezhda Chirkova et al. (Naver Labs Europe), ICLR 2025, https://arxiv.org/abs/2501.16214 [verified]. Finding: sentence-level sequence labeling (DeBERTa, 430M) that also serves as a reranker. Pruning is reported with "negligible to no drop" across domains, and it compares favorably to RECOMP-extractive, DSLR and LLMLingua. Relevance: a further case where sentence-granular extractive pruning beats token pruning. It is query-aware.
- **CompAct**, "CompAct: Compressing Retrieved Documents Actively for Question Answering", Chanwoong Yoon et al., EMNLP 2024, https://arxiv.org/abs/2407.09014 [prior knowledge]. Finding: iterative abstractive compression for multi-doc QA, with high compression (~47x reported) and better multi-hop QA than RECOMP. Relevance: a case where abstractive compression wins, but it is query-aware and multi-hop. It is a weak counter to F1 in our single-doc, question-blind setting.
- **ACoRN**, "Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models", arXiv Dec 2025, https://arxiv.org/abs/2512.08943 [verified exists via index; authors not confirmed]. Finding (abstract): abstractive compressors "are more likely to omit important information essential for the correct answer, especially in long contexts". Relevance: omission, not fabrication, is the abstractive failure mode, which is consistent with F1.

### Inferences
- The literature's mechanism for "token pruning hurts QA" (broken grammar/comprehension) is the same mechanism as our anaphora breakage, at a finer granularity. Our F3 is the sentence-to-discourse-level version of a known token-level effect.
- Abstractive compression's documented failure is omission of answer-critical details. That likely explains why our paraphrased/structured claims lost to verbatim text (F1).

### Gaps
- I found no head-to-head study measuring hallucination or abstention rates for extractive vs abstractive compressors under the same budget.
- I found no paper reporting *abstention* (as opposed to wrong answers or hallucination) as a consequence of fragmented compressed context. Our F3 abstention observation appears to be unreported.

## Q3. Fixed-ratio evaluations (e.g. 2x) on QA, and coreference/anaphora breakage

### Takeaway
Fixed-ratio QA evaluations exist (LongBench/ZeroSCROLLS at 2x–10x, GACC-Bench at 5–20%). At mild ratios (~2x) extractive methods are near-lossless or even improve accuracy. No compression paper I found measures coreference breakage directly. The only adjacent evidence comes from coreference-resolution-in-RAG/long-context papers, which show that unresolved references in context hurt QA.

### Cited Findings
- **Jha et al. 2024** (https://arxiv.org/abs/2407.08892): ratio sweeps on LongBench. Larger chunks (up to 512 tokens) do *worse* at high compression because fewer fit, which trades coherence for coverage. Relevance: the direct literature counterpart of F2's coherence-vs-coverage tension; at high ratios they found coverage wins, while at our ~2x we found coherence wins.
- **Zhang et al. 2025** (https://arxiv.org/abs/2505.00019): "moderate compression even enhances LLM performance" on LongBench. Relevance: consistent with mild (~2x) compression being near-free.
- **LLMLingua-2** (https://arxiv.org/abs/2403.12968) [prior knowledge]: evaluated at 2x–5x on MeetingBank QA/summarization, LongBench and ZeroSCROLLS. Relevance: the standard 2x reference point for a task-agnostic baseline.
- **GACC-Bench 2026** (https://arxiv.org/abs/2610.21893): fixed 5%/10% retention with scaling to 20%. LLM compressor output is truncated to budget by keeping the first tokens, which can cut mid-thought; coherence is not measured. Relevance: a fixed-budget, question-blind protocol close to ours, but far more aggressive.
- **Jang et al.**, "From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on Retrieval-Augmented Generation systems", Youngjoon Jang et al. (Korea Univ.), ACL 2025 SRW, https://arxiv.org/abs/2507.07847 / https://aclanthology.org/2025.acl-srw.27 [verified]. Finding: coreferential complexity in retrieved documents "introduces ambiguity that disrupts in-context learning". Resolving coreference improves both retrieval and QA. Relevance: the closest evidence that unresolved anaphora (our dropped antecedents, F3) degrades reader QA. A coref-resolve-then-select pipeline is a known mitigation.
- **LQCA**, "Bridging Context Gaps: Leveraging Coreference Resolution for Long Contextual Understanding", arXiv Oct 2024 (ICLR 2025 per prior knowledge), https://arxiv.org/abs/2410.01671 [verified exists via search; venue prior knowledge]. Finding: resolving coreference across long-context partitions before reading improves long-context QA. Relevance: supports rewriting pronouns to explicit antecedents in kept sentences, as a cheaper alternative to keeping all connective text.

### Inferences
- A practical fix that is consistent with both lines of evidence: select verbatim sentences, but run coreference resolution first (or keep the antecedent sentence), so kept sentences remain self-contained.
- Our measured trade-off (coherent contiguous text beats max-coverage at ~45%) may flip at harsher budgets, per Jha et al.'s chunk-size result.

### Gaps
- I found no compression paper (token, sentence or abstractive) that measures coreference/anaphora breakage as an outcome. This appears to be an open gap that our F3 result addresses.
- No paper I found evaluates on RepLiQA specifically.

## Q4. Soft/learned compression (brief contrast)

### Takeaway
Learned soft compressors (gist tokens, ICAE, xRAG, Cartridges) can be near-lossless on RAG and long-doc QA but fail on exact-detail recall. They need model access or training, so they are not usable with a black-box reader like Haiku.

### Cited Findings
- **Gist tokens**, "Learning to Compress Prompts with Gist Tokens", Jesse Mu et al., NeurIPS 2023, https://arxiv.org/abs/2304.08467 [prior knowledge]. Finding: compresses instructions into a few gist tokens (up to 26x) with small loss.
- **ICAE**, "In-context Autoencoder for Context Compression in a Large Language Model", Tao Ge et al., ICLR 2024, https://arxiv.org/abs/2307.06945 [prior knowledge]. Finding: an autoencoder compresses context into memory slots (~4x).
- **xRAG**, "xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token", Xin Cheng et al., NeurIPS 2024, https://arxiv.org/abs/2405.13792 [prior knowledge]. Finding: projects a retrieval embedding into a single token.
- **Deng et al.**, "A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression", Chenlong Deng et al., ACL 2025, https://arxiv.org/abs/2412.17483 [verified]. Finding: only slight loss on RAG and long-doc QA, but clear failures on synthetic recall and reranking. Named failure modes are "lost by the boundary", "lost if surprise" and "lost along the way", with the bottleneck located in the gist representations. Relevance: soft compression loses exactly the verbatim details that RepLiQA-style questions probe, which supports F1 from another angle.
- **Cartridges**, "Cartridges: Lightweight and general-purpose long context representations via self-study", Sabri Eyuboglu et al. (Stanford), arXiv June 2025, https://arxiv.org/abs/2506.06266 [verified]. Finding: trains a per-corpus KV cache offline (query-agnostic, amortized) with synthetic self-study conversations. It matches ICL quality at 38.6x less memory and 26.4x higher throughput. Relevance: a learned equivalent of "compile the corpus before questions". It needs white-box training, so it is not an option with a Haiku reader.
- **Feldman & Artzi**, "Simple Context Compression: Mean-Pooling and Multi-Ratio Training", arXiv Oct 2025, https://arxiv.org/abs/2510.20797 [verified exists via search]. Finding: mean-pooling soft compression beats the common compression-token approach.

### Inferences
- Soft methods are a contrast only. Their documented weakness is detail recall, while their strength is aggregate or gist tasks.

### Gaps
- I found no evaluation of soft compressors on RepLiQA or with black-box API readers (they can't be applied to them).

## Q5. Benchmarks and surveys comparing compression methods

### Takeaway
The useful comparative studies are Jha et al. 2024 (extractive > abstractive > token pruning on long-context QA), Zhang et al. 2025 (six methods; hallucination rises with compression), Łajewska et al. 2025 (information preservation), GACC-Bench 2026 (question-blind and goal-conditioned budgets) and the NAACL 2025 survey.

### Cited Findings
- **Survey**, "Prompt Compression for Large Language Models: A Survey", Zongqian Li et al., NAACL 2025, https://arxiv.org/abs/2410.12388 [prior knowledge]. Content: a hard vs soft prompt taxonomy; covers the LLMLingua family, Selective Context, RECOMP, gist/ICAE/xRAG and open challenges.
- **Survey**, "Efficient Prompting Methods for Large Language Models: A Survey", arXiv 2024, https://arxiv.org/abs/2404.01077 [verified exists via search]. Content: frames pruning as removing lexical units "without changing the linguistic expression", contrasted with abstractive rewriting.
- **Comparative studies**: Jha et al. 2024 (https://arxiv.org/abs/2407.08892), Zhang et al. 2025 (https://arxiv.org/abs/2505.00019), Łajewska et al. 2025 (https://arxiv.org/abs/2503.19114) and GACC-Bench 2026 (https://arxiv.org/abs/2610.21893); details are under Q1–Q3.
- **Recent query-aware long-context compressors** found in 2026 searches, as contrast only: "LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning" (https://arxiv.org/abs/2606.01336), "TopoCompress: Long Context Compression via Graph-Wired Semantic Trajectories" (https://arxiv.org/abs/2608.30811) and "End-to-End Context Compression at Scale" (https://arxiv.org/abs/2606.09659, which uses KVzip as its query-agnostic baseline) [exist per search index; not read, findings unverified].

### Inferences
- Taken together, the benchmarks support our design direction. Mild question-blind compression is close to free if it is extractive and sentence-coherent. Paraphrase loses details and token pruning breaks comprehension.

### Gaps
- No benchmark isolates "discourse coherence of the compressed context" as a variable. No study reports abstention rate vs compression method. These are both places where our results add something new.
- PCToolkit and other toolkit papers were not checked.
