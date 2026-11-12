# Offline LLM-built structured corpus representations for QA ("compile once, query many") — state of the art 2024–2026, as it bears on WiCER

Verification legend used on every item:
- **[V-abs]** = arXiv abstract/metadata pulled directly from the arXiv API this session (title, authors, date, abstract text confirmed).
- **[V-body]** = paper body (HTML) also opened; numbers taken from tables/text.
- **[2nd]** = secondhand (search snippet, blog, aggregator, or prior knowledge not re-opened). Treat numbers as unconfirmed.

All arXiv ids below were confirmed by the arXiv API unless marked [2nd].

---

## Q1. The structured-index family (GraphRAG, LightRAG, HippoRAG 1/2, RAPTOR, KAG, MemoRAG, PathRAG, NodeRAG, StructRAG, SiReRAG, ArchRAG and 2025–2026 successors): method, indexing cost, gains

### Takeaway
Almost every method in this family is still *retrieval at query time over an LLM-built index* (graph, tree, community summaries); only a handful put a compiled artifact directly in context. Reported gains are concentrated on multi-hop QA (single-digit F1 points; e.g., SiReRAG +1.9 F1 avg, KAG +19.6%/+33.5% relative F1) and on global "sensemaking" questions judged by LLM pairwise win rates; the original GraphRAG indexing is expensive (LazyGraphRAG: its own indexing is 0.1% of full GraphRAG's), and the most recent methods (HippoRAG 2, LinearRAG, KET-RAG, LeanRAG) are mainly about cutting cost and reattaching raw passages to the structure.

### Cited Findings

**Foundational (2024)**
- **GraphRAG** — Edge, Trinh, Cheng, Bradley, Chao, Mody et al. (Microsoft), "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", arXiv 2404.16130, Apr 2024. [V-abs] Method: an LLM extracts an entity KG from chunks, then *pregenerates community summaries* for clusters of related entities (Leiden hierarchy). At query time, every community summary produces a partial answer, and those partial answers are reduced into a final answer (map-reduce). Evaluated on "global sensemaking questions over datasets in the 1 million token range"; claims "substantial improvements over a conventional RAG baseline for both the comprehensiveness and diversity" (LLM-judged, no exact-match accuracy). Relevance: the community-summary layer is the closest mainstream analogue to a compiled wiki, but it is aimed at global questions, not detail recall. — [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)
- **LazyGraphRAG** (Microsoft Research blog, Nov 2024; no paper). Defers all LLM summarization to query time. Indexing cost is "identical to vector RAG and 0.1% of the costs of full GraphRAG". At 4% of GraphRAG global-search query cost it "significantly outperforms" competing methods on local and global queries, measured on 5,590 AP news articles. [2nd: blog via search snippet] — [MSR blog](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- **LightRAG** — Guo, Xia, Yu, Ao, Huang (HKU), arXiv 2410.05779, Oct 2024. [V-abs] Method: LLM entity and relation extraction into a graph plus vectors, with "dual-level retrieval" (low-level entities and high-level themes) and incremental updates without a full rebuild. Claims improvements in retrieval accuracy and efficiency. I did not open the headline win rates (UltraDomain LLM-judge pairwise). The widely repeated "6,000x fewer tokens" figure is about *query-time* retrieval tokens versus GraphRAG global search, not indexing [2nd]. — [arXiv 2410.05779](https://arxiv.org/abs/2410.05779), [search summary](https://futureagi.com/blog/graphrag-vs-rag/)
- **HippoRAG** — Jiménez Gutiérrez, Shu, Gu, Yasunaga, Su (OSU), NeurIPS 2024, arXiv 2405.14831. [V-abs] Method: OpenIE triples become a KG, and Personalized PageRank runs from query entities ("hippocampal index"). Up to 20% better on multi-hop QA. Single-step retrieval matches or beats iterative IRCoT while being "10–30 times cheaper and 6–13 times faster". — [arXiv 2405.14831](https://arxiv.org/abs/2405.14831)
- **RAPTOR** — Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning (Stanford), arXiv 2401.18059, Jan 2024 (ICLR 2024 [2nd]). [V-abs] Method: recursively embed, cluster, and summarize chunks into a tree of summaries, then retrieve across levels. With GPT-4, it improves the best QuALITY result "by 20% in absolute accuracy". Relevance: a query-agnostic hierarchical summary is the closest classic analogue of a compiled wiki, *but leaf chunks are kept and retrievable*. — [arXiv 2401.18059](https://arxiv.org/abs/2401.18059)
- **KAG** — Liang, Sun, Gui, Zhu, Jiang, Zhong et al. (Ant Group), arXiv 2409.13731, Sep 2024. [V-abs] Method: LLM-friendly knowledge representation, *mutual indexing between KG and original chunks*, and a logical-form-guided hybrid reasoner. Relative F1 gains: +19.6% on 2Wiki and +33.5% on HotpotQA. Deployed for e-government and e-health QA. — [arXiv 2409.13731](https://arxiv.org/abs/2409.13731)
- **MemoRAG** — Qian, Liu, Zhang, Mao, Lian, Dou et al., TheWebConf 2025, arXiv 2409.05591. [V-abs] Method: a light long-range model builds a *global memory* of the whole corpus (implemented as KV compression and trained with RLGF). Given a task, the memory generates draft answers or "clues", which then drive retrieval for a stronger generator. Claims superiority on varied long-context tasks. Relevance: the closest to WiCER's "compiled memory as context" idea, but memory is used to *guide retrieval*, not as the final context. — [arXiv 2409.05591](https://arxiv.org/abs/2409.05591)
- **StructRAG** — Li, Chen, Yu, Lin, Lu, Tang et al., arXiv 2410.08815, Oct 2024 (ICLR 2025 [2nd]). [V-abs] Method: a router picks the best structure type for the task (table, graph, algorithm, catalogue, chunk). The documents are then rebuilt in that structure *at inference time*, and the answer is inferred from it. Reports SOTA on knowledge-intensive reasoning (Loong, Podcast transcripts [2nd]). Relevance: evidence that the structure type should match the question type. It is query-time, though, not query-agnostic. — [arXiv 2410.08815](https://arxiv.org/abs/2410.08815)
- **SiReRAG** — Zhang, Choubey, Fabbri, Bernadett-Shapiro, Zhang, Mitra et al. (Salesforce), ICLR 2025, arXiv 2412.06206. [V-abs] Method: a *similarity tree* (RAPTOR-like recursive summaries) plus a *relatedness tree* (propositions grouped by shared entities, then summarized), flattened into one retrieval pool. Average F1 is +1.9 on MuSiQue, 2Wiki, and HotpotQA, and up to +7.8 F1 with rerankers. Relevance: propositions are the unit that preserves detail facts. — [arXiv 2412.06206](https://arxiv.org/abs/2412.06206)

**2025 successors**
- **HippoRAG 2** — Jiménez Gutiérrez, Shu, Qi, Zhou, Su, ICML 2025, arXiv 2502.14802. [V-abs] Explicitly diagnoses that structure-augmented RAG "performance on more basic factual memory tasks drops considerably below standard RAG". The fix is deeper *passage integration* (passages become graph nodes alongside phrases) plus online LLM filtering. Reports +7% on associative-memory tasks over a SOTA embedding model, while not losing on factual or sensemaking tasks. — [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- **PathRAG** — Chen, Guo, Yang, Chen, Chen, Liu et al., arXiv 2502.14902, Feb 2025. [V-abs] Argues graph-RAG's problem is *redundancy*, not insufficiency. It uses flow-based pruning to extract key relational paths and renders them as text in the prompt. Claims consistent wins on six datasets across five (LLM-judged) dimensions. — [arXiv 2502.14902](https://arxiv.org/abs/2502.14902)
- **NodeRAG** — Xu, Zheng, Li, Chen, Liu, Chen et al., arXiv 2504.11544, Apr 2025. [V-abs] A heterogeneous graph with typed nodes (entities, relations, semantic units, attributes, high-level summaries, original text [2nd for node list]). Beats GraphRAG and LightRAG on indexing time, query time, storage, multi-hop QA, and head-to-head judging, "with minimal retrieval tokens". — [arXiv 2504.11544](https://arxiv.org/abs/2504.11544)
- **ArchRAG** — Wang, Fang, Zhou, Liu, Ma, AAAI 2026, arXiv 2502.09891. [V-abs] Attributed communities with LLM-based hierarchical clustering and a hierarchical index. Claims better accuracy *and* lower token cost than GraphRAG-style systems. — [arXiv 2502.09891](https://arxiv.org/abs/2502.09891)
- **KET-RAG** — Huang, Zhang, Xiao, arXiv 2502.09304, Feb 2025. [V-abs] Builds the LLM KG "skeleton" only on a small set of *key chunks* and covers the remaining chunks with a cheap text–keyword bipartite graph. Retrieval quality matches or beats MS GraphRAG while "reducing indexing costs by over an order of magnitude", or reaches up to +32.4% generation quality at about 20% lower indexing cost. Evaluated against 13 baselines. Relevance: selective, budgeted LLM compilation, analogous to WiCER's budget. — [arXiv 2502.09304](https://arxiv.org/abs/2502.09304)
- **HiRAG** (Huang et al., EMNLP 2025 Findings, arXiv 2503.10150) [V-abs], **LeanRAG** (Zhang et al., AAAI-26, arXiv 2508.10391; −46% retrieval redundancy) [V-abs], **HyperGraphRAG** (Luo et al., NeurIPS 2025, arXiv 2503.21322; n-ary hyperedges) [V-abs], **GFM-RAG** (Luo et al., NeurIPS 2025, arXiv 2502.01113; an 8M-param graph foundation model retriever trained on 60 KGs and 700k docs) [V-abs], and **Youtu-GraphRAG** (Dong et al., arXiv 2508.19855; schema-bounded extraction, up to 90.71% token saving and +16.62% accuracy; introduces an *anonymized* dataset to control for pretraining knowledge leakage) [V-abs]. — [2503.10150](https://arxiv.org/abs/2503.10150), [2508.10391](https://arxiv.org/abs/2508.10391), [2503.21322](https://arxiv.org/abs/2503.21322), [2502.01113](https://arxiv.org/abs/2502.01113), [2508.19855](https://arxiv.org/abs/2508.19855)
- **LinearRAG** — Zhuang, Chen, Xiao, Zhou, Zhang, Chen et al. (PolyU), arXiv 2510.10114, Oct 2025. [V-abs] Drops LLM relation extraction entirely ("unstable and costly… noisy graphs"). It builds a relation-free "Tri-Graph" from lightweight entity extraction plus semantic links, which "scales linearly with corpus size and incurs no extra token consumption". Claims significant gains over GraphRAG baselines on four datasets. — [arXiv 2510.10114](https://arxiv.org/abs/2510.10114)
- **RAGU** — Komarov, Bondarenko et al., arXiv 2607.11683, Jul 2026. [V-abs] Extraction is separated from consolidation, with a 7B "Meno-Lite" extractor. On GraphRAG-Bench Medical, evidence recall reaches 0.84 vs ≤0.76. It shows that the "apparent HippoRAG2 advantage" on multi-hop factoid QA "is largely an answer-format artifact". Cost appendix [2nd via snippet]: MS-GraphRAG construction is about 40k tokens per document (about $0.10 per document on gpt-4o, about $10k per 100k docs), versus about 6k tokens per document for HippoRAG 2 on a local model. — [arXiv 2607.11683](https://arxiv.org/abs/2607.11683)
- Other indexing-cost datapoints [2nd, conflicting by setup]: FG-RAG (arXiv 2504.07103 [V-abs]) reports about 7.7M indexing tokens for GraphRAG, 5.1M for LightRAG, and 4.9M for FG-RAG on its Mix dataset. — [FG-RAG](https://arxiv.org/pdf/2504.07103)

**Agentic / "reading-agent" compiled memories (closest to a wiki)**
- **MemWalker** — Chen, Pasunuru, Weston, Celikyilmaz, arXiv 2310.05029 (2023). [V-abs] Builds a tree of summary nodes over a long text, and the LLM navigates it per query. Beats long-context, recurrence, and retrieval baselines on long-text QA. — [arXiv 2310.05029](https://arxiv.org/abs/2310.05029)
- **ReadAgent** — Lee, Chen, Furuta, Canny, Fischer (Google DeepMind), arXiv 2402.09727, Feb 2024 (ICML 2024 [2nd]). [V-abs] Compresses episodes into *gist memories*, then *looks up the original passages* when details are needed. Extends effective context 3.5–20x and beats retrieval, full-context, and gist-only baselines on QuALITY, NarrativeQA, and QMSum. Relevance: a gist-only baseline exists and loses to gist plus lookup, which is the same detail-loss phenomenon WiCER sees. — [arXiv 2402.09727](https://arxiv.org/abs/2402.09727)
- **GraphReader** — Li, He, Guo et al., EMNLP 2024, arXiv 2406.14550. [V-abs] Structures the long text into a graph of key elements and atomic facts, which an agent explores. With a 4k window it beats GPT-4-128k on LV-Eval from 16k to 256k. — [arXiv 2406.14550](https://arxiv.org/abs/2406.14550)

**2026 "LLM Wiki" line (directly comparable to WiCER)**
- **Karpathy "LLM Wiki" idea file** (GitHub gist, ~Apr 2026). An LLM maintains a markdown wiki of summaries, entity pages, and concept pages over immutable raw sources, governed by a schema file (CLAUDE.md/AGENTS.md) and periodic "health checks". [2nd: secondary write-ups; primary gist not opened] — [VentureBeat](https://venturebeat.com/data/karpathy-shares-llm-knowledge-base-architecture-that-bypasses-rag-with-an), gist referenced as gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- **WiCER** — Huerta, arXiv 2605.07068, May 2026. [V-abs] (the project's own paper) On 17 RepLiQA domains (6,800 Qs), full-context KV inference beats RAG on curated knowledge (4.38 vs 4.08 out of 5). Blind compilation fails, scoring 2.14–2.32 vs 3.46 with 53–60% catastrophic failures. One or two CEGAR-style refine iterations recover 80% of the lost quality, and targeted diagnosis contributes +0.95 vs +0.16 for generic pinning. — [arXiv 2605.07068](https://arxiv.org/abs/2605.07068)
- **Streaming Knowledge Compilation** — Huerta, arXiv 2606.09877, Jun 2026. [V-abs] Same author. Proactive "materiality-scored pinning" for a time-evolving wiki under a fixed token budget, with *queries unknown at ingestion*, and an O(√(T log K)) regret bound. Reports "a pervasive LLM-as-judge confound on post-training knowledge" and argues for regret rather than absolute QA scores. — [arXiv 2606.09877](https://arxiv.org/abs/2606.09877)
- **Single-Round Vector RAG vs an LLM-Compiled Wiki** — Cochran, arXiv 2605.18490, May 2026. [V-abs] Preregistered; 24 papers, 13 questions, two blinded judges. The wiki wins on cross-paper synthesis but spends about 21x more query tokens (about 1.65M vs 78k over 13 Qs [2nd for the exact totals]), so "no break-even point exists". RAG met the single-fact groundedness test (judge-sensitive). Decomposition-RAG closed most of the synthesis gap more cheaply. Judge rank agreement was rho = 0.04 on holistic groundedness vs 0.81 on the most concrete criterion. — [arXiv 2605.18490](https://arxiv.org/abs/2605.18490)
- **Progressive Disclosure for LLM-Maintained Wiki KBs** — Cochran, arXiv 2607.04576, Jul 2026. [V-abs] A 709-page LLM-maintained wiki, with page content held fixed and only the access structure varied (catalog plus one-line summaries). A capable agent never loaded the index. Quality was non-inferior and cost fell, but human–judge agreement was far below plan. — [arXiv 2607.04576](https://arxiv.org/abs/2607.04576)
- **LLM-Wiki (Retrieval as Reasoning)** — Ming, Li, Wu, Que, arXiv 2605.25480, May 2026. [V-abs] Compiles documents into wiki pages with bidirectional links, exposes search, read, and follow-link tools, and keeps an "Error Book" for persistent self-correction. +2.0–8.1 F1 over HippoRAG 2, LightRAG, and GraphRAG on HotpotQA, MuSiQue, and 2Wiki. — [arXiv 2605.25480](https://arxiv.org/abs/2605.25480)
- **Training a Knowledge Base** — Pan, Yu, arXiv 2608.21829, Aug 2026. [V-abs] Treats the store as a non-parametric model *trained on (question, answer) pairs*: a curator agent answers, sees the gold answer, and edits the store. The gain *grows with overlap to the training questions*: parity when no key is shared with training, +0.176 F1 when both keys are shared, and +0.294 F1 on the trained questions themselves. Matches HippoRAG's gains with 1,913 links vs 196,112. Benchmarks are contamination-free (KBGym, PhantomWiki). **This is the closest published analogue of WiCER's probe→held-out gap.** — [arXiv 2608.21829](https://arxiv.org/abs/2608.21829)

### Inferences
- The mainstream family solves a different problem from WiCER: it still retrieves at query time, so detail facts survive in the leaf chunks. The methods that kept raw passages reachable (RAPTOR leaves, KAG mutual indexing, HippoRAG 2 passage nodes, ReadAgent lookups) are exactly the ones that avoided the detail-QA regression. WiCER has no leaf layer, so its compiled wiki has to *be* the leaf layer for any fact that might be asked. That is consistent with the project finding that verbatim rendering beats truncation.
- The cost-reduction trend (LazyGraphRAG, KET-RAG, LinearRAG) says LLM summarization at index time is the expensive *and* noisy part. KET-RAG's "LLM on key chunks only, cheap structure elsewhere" maps cleanly onto "compile summaries for some topics, verbatim render for others".
- Pan & Yu (2608.21829) independently reproduce WiCER's pattern: question-supervised curation helps strongly on trained questions, weakly on near neighbours, and not at all on unrelated ones. The field has no published fix yet beyond "cover more of the corpus".

### Gaps
- I did not open LightRAG's, NodeRAG's, PathRAG's, or ArchRAG's result tables, so exact win rates and accuracies are not recorded. Most use LLM-judge pairwise win rates on UltraDomain-style corpora, which the critique papers in Q2 show are biased.
- I could not find a primary-source dollar figure for MS GraphRAG indexing. The ~40k tokens per document (RAGU appendix) is a search snippet only.
- MemoRAG headline numbers were not opened.

---

## Q2. Independent comparisons and critiques: when does structure help (global/sensemaking vs local/detail)?

### Takeaway
Independent, controlled studies (2025) agree: **plain chunk RAG wins on single-hop and detail questions and on abstention; graph or summary structure wins on multi-hop, comparison, temporal, and summarization questions.** Community-summary ("global") GraphRAG is the worst on detail and null questions. LLM-judge pairwise evaluations, which most GraphRAG papers use, inflate the gains.

### Cited Findings
- **RAG vs. GraphRAG: A Systematic Evaluation and Key Insights** — Han, Ma, Wang, Shomer, Lei, Qi et al., arXiv 2502.11371, Feb 2025. [V-body] Unified protocol; NQ, HotpotQA, MultiHop-RAG, and NovelQA, plus summarization on SQuALITY, QMSum, and ODSum. Results:
  - **Single-hop**: NQ F1 is RAG 64.78, Community-GraphRAG Local 63.01, HippoRAG2 61.03, Community Global 54.48, KG triplets-only 34.28.
  - **Detail questions (NovelQA "dtl")**: RAG 55.28, Community Local 46.88, Community **Global 30.89**.
  - **Multi-hop**: HotpotQA F1 is HippoRAG2 63.01 vs RAG 60.04; MultiHop-RAG accuracy is HippoRAG2 70.27 vs RAG 67.02.
  - **MultiHop-RAG by question type**: temporal is Global 53.34 vs RAG 30.70; comparison is Global 64.02 vs RAG 57.59; inference is RAG 92.16 vs Local 86.89; **null/unanswerable is RAG 96.01, Local 80.07, Global 19.27**.
  - **Answer entities missing from the LLM-built KG**: about 34% on both HotpotQA (65.8% present) and NQ (65.5% present). Triplet-only retrieval accuracy is 39.2% / 32.2% vs RAG's 88.6% / 86.7%.
  - **Summarization**: methods that retrieve original chunks (RAG, RAPTOR, HippoRAG2) do well.
  - **LLM-as-judge**: swapping presentation order can flip verdicts, especially RAG vs GraphRAG-Local.
  - **Integration**: running both RAG and GraphRAG retrieval gives +6.4% on MultiHop-RAG (Llama-3.1-70B); selection gives +1.1%.
  - **Indexing time**: 135 s for RAG vs 5,560 s for Community-GraphRAG and 7,702 s for KG-GraphRAG.
  - **Graph builder quality matters**: MultiHop-RAG accuracy is 65.77 for RAG, 71.17 with a GPT-4o-mini graph, and 75.08 with a GPT-4o graph.
  — [arXiv 2502.11371](https://arxiv.org/abs/2502.11371)
- **When to use Graphs in RAG (GraphRAG-Bench, PolyU)** — Xiang, Wu, Zhang, Chen, Hong, Huang et al., arXiv 2506.05690, Jun 2025. [V-body] Opens with "GraphRAG frequently underperforms vanilla RAG on many real-world tasks". Four task levels; GPT-4o-mini, Table 2:
  - **Fact retrieval (Novel)**: RAG 60.92 vs best GraphRAG 60.14, so graphs do not help.
  - **Complex reasoning (Novel)**: RAG 42.93 vs HippoRAG2 53.38 / MS-GraphRAG 50.93.
  - **Contextual summarization (Novel)**: RAG 51.30 vs MS-GraphRAG 64.40 / HippoRAG2 64.10, about +13.
  - **Medical**: gains are small (about 2–3 points).
  - **Retrieval**: basic RAG has the best fact-retrieval evidence recall (83.21).
  - **Query prompt tokens**: RAG about 879, HippoRAG2 about 1,008, RAPTOR about 3,441, MS-GraphRAG local about 38.7k, LightRAG about 100.8k, MS-GraphRAG global about 331k.
  - **Guideline**: use plain RAG for fact lookup and graphs for reasoning and summarization; prefer lightweight graphs (HippoRAG2) over global community summaries.
  — [arXiv 2506.05690](https://arxiv.org/abs/2506.05690), [repo](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- **GraphRAG-Bench (domain reasoning)** — Xiao, Dong, Zhou et al., arXiv 2506.02404, Jun 2025. [V-abs] A different benchmark with the same name: college-level questions from 20 textbooks across 16 disciplines; evaluates nine GraphRAG methods across construction, retrieval, and generation. — [arXiv 2506.02404](https://arxiv.org/abs/2506.02404)
- **In-depth Analysis of Graph-based RAG in a Unified Framework** — Zhou, Su, Sun, Wang et al., arXiv 2503.04338, Mar 2025. [V-abs] Unified comparison over "specific questions to abstract questions". New combined variants beat SOTA on each, which implies no single structure wins both specific and abstract questions. — [arXiv 2503.04338](https://arxiv.org/abs/2503.04338)
- **How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG** — Zeng, Luo, Lin et al., arXiv 2506.06331, May 2025. [V-abs] Identifies "unrelated questions and evaluation biases" in GraphRAG evaluation. Under graph-text-grounded questions and debiased judging, the gains of three representative GraphRAG methods are "much more moderate than reported previously". — [arXiv 2506.06331](https://arxiv.org/abs/2506.06331)
- **HippoRAG 2** (above) concedes that structure-augmented RAG degrades basic factual memory below standard RAG [V-abs]. **RAGU** shows a HippoRAG2 multi-hop advantage is an answer-format artifact [V-abs]. **Youtu-GraphRAG** builds an anonymized dataset because pretraining knowledge leaks into GraphRAG scores [V-abs]. — [2502.14802](https://arxiv.org/abs/2502.14802), [2607.11683](https://arxiv.org/abs/2607.11683), [2508.19855](https://arxiv.org/abs/2508.19855)
- **Cochran 2605.18490** (above): the compiled wiki wins synthesis, RAG wins single-fact groundedness, and the winner flips with baseline, scorer, and judge. — [arXiv 2605.18490](https://arxiv.org/abs/2605.18490)

### Inferences
- The dividing line the field has converged on is **local/detail/unanswerable → raw chunks; global/multi-hop/temporal/comparison → structure**. WiCER's held-out drop and its losses on detail facts are the expected signature of a summary-type structure without a raw-text fallback. The "unanswerable worse" result from the TACT milestone matches Han et al.'s null-question collapse for community summaries (96 → 19).
- Han et al.'s finding that about 34% of answer entities never make it into the LLM-built KG is the graph-world equivalent of WiCER's "compilation gap". It is a direct, citable estimate of how much query-agnostic LLM extraction drops on QA-relevant facts.
- Pairwise LLM-judge win rates (GraphRAG, LightRAG, PathRAG, NodeRAG headline numbers) should not be trusted as evidence for detail QA. Reference-based, order-swapped, or human-checked scoring is the norm in the critiques.

### Gaps
- No study I found isolates *in-context* compiled structure (no retrieval) versus raw chunks on detail QA at fixed token budget. That is WiCER's exact setting, and the critiques all retain retrieval.
- I did not extract the position-bias magnitudes (figure only in Han et al.).

---

## Q3. Cache-augmented generation, long-context vs RAG, and "knowledge compilation into context or weights"

### Takeaway
When the corpus fits, long context generally beats chunk RAG for strong models, but **RAG wins for small and open models at long lengths and on hallucination/unanswerable detection** (LaRA). **Summary-based retrieval roughly matches long context while chunk retrieval lags** (Li et al. 2024/25). The strongest "compile into a compact artifact" results come from *trained* artifacts (Cartridges, EntiGraph, Knowledge Modules), all of which use **synthetic self-generated questions or conversations** as the compilation signal.

### Cited Findings
- **CAG: Don't Do RAG** — Chan, Chen, Cheng, Huang, WWW'25 short, arXiv 2412.15605. [V-abs] Preloads the whole (small) knowledge base into context and caches the KV, with no retrieval. Reports "comparable or superior results" with "reduced complexity" when the KB is "of a limited and manageable size". — [arXiv 2412.15605](https://arxiv.org/abs/2412.15605)
- **LaRA** — Li, Zhang, Jiang, Xie, Huang, Wang et al. (Alibaba), arXiv 2502.09977, Feb 2025 (ICML 2025 [2nd]). [V-body] 2,326 cases across novels, financial statements, and papers, at 32k and 128k.
  - **LC-minus-RAG gap by task**: location +0.08 / −5.25; reasoning +4.66 / −1.39; comparison +15.22 / +14.30; **hallucination detection −10.38 / −22.36** (RAG better).
  - **Across models**: averaged, LC leads by 2.4 points at 32k, but RAG leads by 3.68 at 128k. At 128k, RAG beats LC by 6.48 on Llama-3.2-3B and by 38.12 on Mistral-Nemo-12B. "RAG provides more significant improvements for weaker models."
  - **Chunk count**: the 7B model peaks at an intermediate chunk count, while the 72B model keeps improving.
  - **Position**: lost-in-the-middle hurts LC.
  — [arXiv 2502.09977](https://arxiv.org/abs/2502.09977)
- **RAG or Long-Context LLMs? (Self-Route)** — Li, Li, Zhang, Mei, Bendersky (Google), EMNLP 2024 industry, arXiv 2407.16833. [V-abs] When sufficiently resourced, LC beats RAG on average, but RAG is far cheaper. Self-Route routes by model self-reflection and gets comparable quality at lower cost. — [arXiv 2407.16833](https://arxiv.org/abs/2407.16833)
- **Long Context vs. RAG for LLMs: An Evaluation and Revisits** — Li, Cao, Ma, Sun (NTU), arXiv 2501.01880, Dec 2024. [V-abs] After filtering questions answerable without context, LC generally beats RAG, especially on Wikipedia-based QA. "**Summarization-based retrieval performs comparably to LC, while chunk-based retrieval lags behind**", and RAG is better for dialogue and general queries. — [arXiv 2501.01880](https://arxiv.org/abs/2501.01880)
- **In Defense of RAG (OP-RAG)** — Yu, Xu, Akkiraju (NVIDIA), arXiv 2409.01666. [V-abs] Order-preserving RAG shows an inverted-U over chunk count, with sweet spots that beat full LC using far fewer tokens. — [arXiv 2409.01666](https://arxiv.org/abs/2409.01666)
- **Long-Context LLMs Meet RAG** — Jin, Yoon, Han, Arik (Google), arXiv 2410.05983. [V-abs] Quality first rises and then falls as passages are added. Hard negatives are the cause, and reordering mitigates it. — [arXiv 2410.05983](https://arxiv.org/abs/2410.05983)
- **Summary of a Haystack (SummHay)** — Laban, Fabbri, Xiong, Wu (Salesforce), arXiv 2407.01370. [V-abs] Without a retriever, long-context GPT-4o and Claude 3 Opus score below 20% on the joint coverage+citation score. Even oracle-retrieval systems trail the human estimate (56%) by 10+ points. — [arXiv 2407.01370](https://arxiv.org/abs/2407.01370)
- **Cartridges (self-study)** — Eyuboglu, Ehrlich, Arora, Guha, Zinsley, Liu et al. (Stanford), arXiv 2506.06266, Jun 2025. [V-abs] Trains a *small KV cache offline per corpus*. Next-token training on the corpus "is not competitive with ICL". **Self-study** works instead: generate synthetic conversations about the corpus and train with context distillation. Matches ICL with 38.6x less memory and 26.4x higher throughput, extends effective context from 128k to 484k on MTOB, and Cartridges compose at inference. Relevance: the closest trained analogue of "compiled wiki in KV cache". The training signal is self-generated questions, like WiCER's probes. — [arXiv 2506.06266](https://arxiv.org/abs/2506.06266)
- **Synthetic continued pretraining (EntiGraph)** — Yang, Band, Li, Candès, Hashimoto (Stanford), arXiv 2409.07431, Sep 2024 (ICLR 2025 [2nd]). [V-body]
  - Setup: QuALITY corpus of 1.3M tokens (265 docs), expanded to 455M synthetic tokens via entity-pair relation text.
  - Closed-book QuALITY accuracy: Llama-3-8B rises 39.49 → 56.22. GPT-4 closed-book is 51.30.
  - With RAG: EntiGraph-CPT+RAG reaches 62.60 vs base+RAG 60.35.
  - Accuracy scales log-linearly with synthetic tokens. Raw-corpus CPT *hurts*.
  - Relevance: knowledge must be "rearranged" into many diverse restatements before a model can use it.
  — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- **Knowledge Modules / Deep Context Distillation** — Caccia, Ansell, Ponti, Vulić, Sordoni, COLM 2025, arXiv 2503.08727. [V-abs] Per-document LoRA. Next-token prediction is poor, and distilling a teacher that has the document in context works better. KMs are synergistic with RAG. — [arXiv 2503.08727](https://arxiv.org/abs/2503.08727)
- **KBLaM** — Wang, Isazawa, Mikaelyan, Hensman (Microsoft), arXiv 2410.10450. [V-abs] KB triples become KV vectors injected via rectangular attention, with no retrieval. More than 10k triples fit into an 8B model with an 8K context on one A100. — [arXiv 2410.10450](https://arxiv.org/abs/2410.10450)

### Inferences
- WiCER's setting is "CAG over a compiled wiki with a small model". LaRA's small-model results warn that small models are the ones for which putting more in context hurts most (RAG beats LC for 3B–12B at 128k), and they lose most on hallucination detection. That fits WiCER's worse unanswerable behaviour and argues for a tight, verbatim-heavy wiki.
- Li et al. 2501.01880 ("summarization-based retrieval ≈ LC") is the best external support that a compiled summary layer *can* match raw text, but there the summaries are still retrieved per query with chunks available.
- Cartridges, EntiGraph, and DCD all find that naively compressing the raw corpus fails and that **query-like synthetic data is what makes compilation work**. They generate many diverse questions (thousands of conversations, 455M tokens), not a small probe set. WiCER's limited probe set may simply undersample the query distribution, which would explain the held-out drop (cf. Pan & Yu's coverage→generalization result).

### Gaps
- I found no paper directly testing CAG over an *LLM-compiled summary* versus CAG over raw text at matched tokens with a small reader, other than WiCER itself and the 2026 Cochran papers (which use an agent with tools, not a fixed context).
- Cartridges' per-benchmark accuracy numbers were not extracted.

---

## Q4. Query-aware vs query-agnostic indexing; synthetic questions at index time (doc2query, HyQE, QuOTE, question-supervised curation)

### Takeaway
Generating questions per chunk at index time reliably improves *retrieval recall* (doc2query lineage, QuOTE, atomic-unit questions, QuIM-RAG). Hallucinated or low-quality generated queries hurt and should be filtered (Doc2Query--). The only work that measures generalization from *training* questions to *unseen* questions on a curated store (Pan & Yu 2026) finds the benefit decays with distance from the training questions. That mirrors WiCER's probe vs held-out gap.

### Cited Findings
- **Doc2Query--: When Less is More** — Gospodinov, MacAvaney, Macdonald, ECIR 2023, arXiv 2301.03266. [V-abs] Doc2Query expansion "is indeed prone to hallucination", which harms effectiveness and inflates the index. Filtering generated queries with a relevance model gives up to +16% effectiveness, −23% query time, and −33% index size. (Original doc2query: Nogueira et al. 2019 [2nd].) — [arXiv 2301.03266](https://arxiv.org/abs/2301.03266)
- **Question-Based Retrieval using Atomic Units for Enterprise RAG** — Raina, Gales (Cambridge), arXiv 2405.12363, May 2024. [V-abs] Decompose chunks into atomic statements, generate synthetic questions per atom, and retrieve by question similarity. Atoms beat chunks on recall, and questions over atoms add a further gain. — [arXiv 2405.12363](https://arxiv.org/abs/2405.12363)
- **Dense X Retrieval (propositions)** — Chen, Wang, Chen, Yu, Ma, Zhao et al., arXiv 2312.06648, Dec 2023 (EMNLP 2024 [2nd]). [V-abs] Indexing by atomic propositions "significantly outperforms passage-level units" and improves downstream QA at a fixed token budget. — [arXiv 2312.06648](https://arxiv.org/abs/2312.06648)
- **QuOTE: Question-Oriented Text Embeddings** — Neeser, Latimer, Khatri, Latimer, Ramakrishnan, arXiv 2502.10976, Feb 2025. [V-abs] Augments chunks with hypothetical questions they answer, which "significantly enhances retrieval accuracy, including in multi-hop". Evaluated on NQ, SQuAD, and MultiHop-RAG against HyDE [2nd]. An "optimal 10–15 questions per chunk" claim appears only in an AI-generated review [2nd, unconfirmed]. — [arXiv 2502.10976](https://arxiv.org/abs/2502.10976)
- **HyQE** — Zhou, Zhang, Hasson, Singh, Li, arXiv 2410.15262, Oct 2024. [V-abs] At ranking time, an LLM hypothesizes the queries that each retrieved context answers, and contexts are ranked by similarity between those hypothetical queries and the user query. Improves ranking without fine-tuning. — [arXiv 2410.15262](https://arxiv.org/abs/2410.15262)
- **QuIM-RAG** — Saha, Saha, Malik, arXiv 2501.02702, Jan 2025. [V-abs] A question-to-question inverted index: generate questions per chunk and match user queries to them. Beats traditional RAG on BERTScore and RAGAS on a custom 500-page corpus (weak evaluation). — [arXiv 2501.02702](https://arxiv.org/abs/2501.02702)
- **Query-aware vs query-agnostic structure**:
  - FG-RAG (arXiv 2504.07103) [V-abs] criticizes GraphRAG's coarse summarization "without being aware of the specific query" and adds query-level fine-grained summarization.
  - StructRAG [V-abs] picks structure per query.
  - MemoRAG [V-abs] uses a query-agnostic global memory to produce query-specific clues.
  - Streaming Knowledge Compilation [V-abs] formalizes compilation with "future queries unknown at ingestion" and uses a query-relevance surrogate for pinning.
  — [2504.07103](https://arxiv.org/abs/2504.07103), [2410.08815](https://arxiv.org/abs/2410.08815), [2409.05591](https://arxiv.org/abs/2409.05591), [2606.09877](https://arxiv.org/abs/2606.09877)
- **Training a Knowledge Base** (Pan & Yu, arXiv 2608.21829) [V-abs]. Gains on held-out questions depend on key overlap with the training questions: none when no key is shared, +0.176 F1 when both keys are shared, and +0.294 on the trained questions. "Accuracy rises steadily with the share of the corpus the indexes cover." — [arXiv 2608.21829](https://arxiv.org/abs/2608.21829)
- **Cartridges self-study and EntiGraph** (Q3): synthetic question and conversation generation over the whole corpus is the compilation signal, and it is generated *broadly* (seeded to cover the corpus) rather than from a few diagnostic probes. [V-abs / V-body]
- Anthropic "Contextual Retrieval" (2024 blog) prepends LLM-generated chunk context before embedding; it is cited by QuOTE as a comparison point. [2nd] — [QuOTE html](https://arxiv.org/html/2502.10976v1)

### Inferences
- Two separable uses of self-generated questions exist in the literature: (a) **as index keys** (doc2query, QuOTE, atomic-unit questions), which help recall but need filtering against hallucination, and (b) **as a training or curation signal** (Cartridges, EntiGraph, Pan & Yu, WiCER's refine loop). For (b), every source that measures it finds that coverage of the corpus, not targeting of the probe set, drives generalization.
- Concrete implications for WiCER, all inferred from the cited work and not tested:
  1. Generate probes to *cover* every document and fact type (proposition- or atom-level, à la Raina & Gales / Dense X) rather than diagnosing a few failures.
  2. Filter probes for answerability (Doc2Query--).
  3. Report held-out results split by overlap with the probes (Pan & Yu's key-overlap buckets), which matches the project's measured "pinned docs +0.10, unpinned −0.13".
  4. Keep a verbatim/proposition fallback for detail facts, as RAPTOR, KAG, HippoRAG 2, and ReadAgent all retain raw text.

### Gaps
- I found no study of index-time question generation used to *decide what goes into a fixed-size in-context artifact*, as opposed to a retrieval index. That is WiCER's specific variant.
- I did not verify QuOTE's or Raina & Gales's numeric recall gains (abstract only).
- HyDE (Gao et al. 2022) and the original doc2query (Nogueira et al. 2019) were not re-opened. I cite them only as lineage [2nd], without ids.
