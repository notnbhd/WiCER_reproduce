# LLM-generated wikis and Wikipedia-style articles (state of the art through Nov 2026), as it bears on compiled-wiki QA (WiCER)

Verification legend: **[V]** = I opened the arXiv abstract / paper HTML / primary page in this session. **[S]** = seen only via search-result snippets or secondary summaries; not opened. **[M]** = from background memory, not re-checked this session.

## 1. Systems that generate wiki articles from sources (STORM lineage and successors)

### Takeaway
The human-reader line (STORM → Co-STORM → OmniThink → WikiAutoGen → deep-research agents) optimizes outline quality, breadth and citation support on open-web topics. None of it measures whether a downstream reader model can answer detail-level questions from the article. Its own evaluations show that 10–15% of generated sentences are unsupported by their citations and that "inferential linking" errors are common.

### Cited Findings
- **[V] STORM**, "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models", Shao et al., NAACL 2024, arXiv 2402.14207. STORM runs pre-writing in two parts: it discovers perspectives, then simulates multi-perspective Q&A against a search-grounded "expert". From this it builds an outline and then writes the article with citations.
  - **FreshWiki** has 100 articles of under 3k words, taken from the most-edited B-class+ Wikipedia pages of Feb 2022–Sep 2023. An average article has 2,159 words, 8.4 sections and 90.1 refs.
  - Outline heading soft recall is 92.73 vs 89.55 for oRAG, and heading entity recall is 45.91 vs 42.38 (GPT-4).
  - Article ROUGE-1 is 45.82 vs 44.26, and article entity recall is 14.10 vs 12.57.
  - Citation recall is 84.83% and citation precision 85.18% (Mistral-7B judge), so about 15% of sentences are unsupported. The top error is "improper inferential linking", followed by "inaccurate paraphrasing" and irrelevant citations.
  - In the human evaluation (10 Wikipedia editors), STORM was preferred 26 vs 14. Verifiability was not better (3.80 vs 3.85). Krippendorff's α was low, 0.22–0.39.
  - Named failure modes are source-bias transfer (7/10 editors), "red herring"/over-speculation, and over-association of unrelated facts. Editors also judged the articles less informative than revised Wikipedia pages.
  - Source: [arXiv HTML](https://arxiv.org/html/2402.14207v2), [abstract](https://arxiv.org/abs/2402.14207).
- **[V] Co-STORM**, "Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations", Jiang, Shao, Ma, Semnani, Lam, EMNLP 2024 (Main), arXiv 2408.15232.
  - Several LM agents converse while the user can steer, and a dynamic mind map organizes the findings into a report.
  - It introduces the **WildSeek** dataset, built from real information-seeking records with user goals.
  - In the human study, 70% of participants preferred it to a search engine and 78% to a RAG chatbot.
  - Source: [arXiv](https://arxiv.org/abs/2408.15232).
- **[V] OmniThink**, "Expanding Knowledge Boundaries in Machine Writing through Thinking", Xi, …, Huajun Chen, Ningyu Zhang (ZJU/Alibaba), EMNLP 2025, arXiv 2501.09751 (v5 Nov 2025).
  - It is a "slow-thinking" writer that iteratively expands and reflects on gathered information before outlining and writing.
  - It argues that vanilla RAG writing "lack[s] depth, novelty, and suffers from redundancy", and claims higher **knowledge density** without loss of coherence or depth.
  - The abstract I opened gives no numbers. Snippets say it beats RAG, oRAG, STORM and Co-STORM on relevance, breadth, depth and novelty. Source: [arXiv](https://arxiv.org/abs/2501.09751).
  - Knowledge density is relevant to WiCER because it is the closest existing "information per token" metric for generated articles. I did not open its formal definition: snippets say unique atomic knowledge units divided by length.
- **[S] WikiAutoGen**, "Towards Multi-Modal Wikipedia-Style Article Generation", arXiv 2503.19065 (Mar 2025).
  - It is a multi-agent system that adds images and multi-perspective self-reflection.
  - It introduces the **WikiSeek** benchmark of under-covered topics with text and images. The authors report +8% to +29% over the prior best (STORM, Co-STORM, OmniThink) on text.
  - Source: [arXiv HTML](https://arxiv.org/html/2503.19065v1), [HF papers](https://huggingface.co/papers/2503.19065).
- **[V] WikiChat**, Semnani, Yao, Zhang, Lam, Findings of EMNLP 2023, arXiv 2305.14292. This is an older chatbot, not an article writer, but it belongs to the Stanford OVAL lineage of STORM.
  - It generates a response, keeps only the facts it can ground in Wikipedia, and adds retrieved facts.
  - Reported factual accuracy is 97.3% in simulated conversations. Against GPT-4 it gains +38.6% on tail knowledge and +51.0% on recent knowledge.
  - It was distilled to LLaMA-7B "with minimal loss of quality". This is an existence proof that a small model can run a grounding pipeline.
  - Source: [arXiv](https://arxiv.org/abs/2305.14292).
- **[S] Deep-research report generators.** DeepResearch Bench (arXiv 2506.11763, Jun 2025; papernotes lists it under ICLR 2026) has 100 PhD-level tasks across 22 domains.
  - **RACE** scores reports against references on comprehensiveness, insight, instruction following and readability.
  - **FACT** extracts statement–URL pairs and reports citation accuracy and effective citations per task.
  - Deep-research agents beat search-augmented LLMs on quality but trade off citation accuracy. Perplexity DR had better citation accuracy than Gemini-2.5-Pro DR and OpenAI DR.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2506.11763), [liner review](https://liner.com/review/deepresearch-bench-comprehensive-benchmark-for-deep-research-agents).
- **[S]** A 2025 survey, "Deep Research: A Systematic Survey" (arXiv 2512.02038), exists. I did not open it. Source: [arXiv PDF](https://arxiv.org/pdf/2512.02038).
- **[M] WikiSum**, Liu et al., "Generating Wikipedia by Summarizing Long Sequences", ICLR 2018. This is seminal multi-document → lead-section generation from cited references plus web search results, evaluated with ROUGE. (From memory; not re-checked.)

### Inferences
- The STORM-family error taxonomy (inferential linking, paraphrase drift, over-association) is the human-facing analogue of WiCER's "compilation gap". The difference is that STORM measures unsupported additions, while WiCER measures dropped facts (omissions). Neither line measures both on the same artifact. That is a citable gap.
- These systems are designed to be *broad* (coverage, novelty) for human readers. They are not designed to preserve exact details (dates, numbers, names) for a downstream QA model. Nothing in this lineage targets verbatim retention, which matches the project's own "Haiku paraphrases" pitfall.

### Gaps
- I did not verify OmniThink's numeric results or its exact knowledge-density formula, or WikiAutoGen's per-metric numbers.
- I found no 2026 successor paper from the Stanford STORM group itself. The repo-level status of STORM/Co-STORM in 2026 was not checked.
- WebGLM (2023) was not checked this session.

## 2. Karpathy's "LLM wiki" idea and papers that formalize or evaluate it

### Takeaway
Karpathy posted the pattern as a gist on 2026-04-04. By November 2026 there is a small, fast-growing arXiv cluster that evaluates it, and WiCER (2605.07068) is one of the first. The other evaluations are small (13–59 questions, single corpora, LLM judges), give mixed results against RAG, and one of them retracted its wiki citation advantage after correcting a scoring bug.

### Cited Findings
- **[V] Karpathy, "LLM Wiki" idea file** (GitHub gist, created 2026-04-04).
  - It has three layers: immutable **raw sources**; an LLM-written and LLM-maintained markdown **wiki** of summaries, entity pages and concept pages; and a **schema** file (CLAUDE.md or AGENTS.md) that defines conventions.
  - It defines three operations. **Ingest**: one source may touch 10–15 pages, and the LLM updates index.md and log.md. **Query**: the LLM answers with citations and can file answers back into the wiki. **Lint**: the LLM checks for contradictions, stale claims, orphan pages and missing links.
  - Its RAG critique is that the LLM "is rediscovering knowledge from scratch on every question". It calls the wiki "a persistent, compounding artifact" and says "Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase."
  - On scale, the index-file approach works "surprisingly well at moderate scale (~100 sources, ~hundreds of pages)".
  - Source: [gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
  - The gist contains no evaluation. Its intended reader is a human plus an agentic LLM that browses with tools, not a small model with the full wiki in context.
- **[V] WiCER**, "Wiki-memory Compile, Evaluate, Refine: Iterative Knowledge Compilation for LLM Wiki Systems", Juan M. Huerta, arXiv 2605.07068, 2026-05-08.
  - The method is a CEGAR-style loop: probe the compiled wiki, find dropped facts, and pin them in recompilation.
  - Evaluation covers 17 RepLiQA domains with 6,800 questions.
  - Full-context KV-cache inference scored 4.38 vs 4.08 for RAG on curated knowledge, with about 7.3× faster TTFT, but it degrades at scale.
  - Blind compilation scored 2.14–2.32 vs a 3.46 baseline, with a 53–60% catastrophic-failure rate.
  - WiCER with 1–2 iterations scored 3.24 vs 3.47 raw, recovering about 80%. Catastrophic failures fell 55% relative. Targeted diagnosis contributed +0.95 vs +0.16 for generic pinning.
  - Source: [arXiv](https://arxiv.org/abs/2605.07068).
- **[V] Cochran**, "Single-Round Vector RAG vs an LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research Corpus", arXiv 2605.18490 (v1 May 18, v3 Oct 2, 2026).
  - Setup: 24 papers and 13 questions. An agent browses the wiki with tools, compared against single-round RAG with the same answer model, scored by two blinded LLM judges.
  - Wiki synthesis looked better but fell below the registered threshold once the judges were pooled. RAG met the registered single-fact groundedness test, though this was judge-sensitive.
  - The wiki was cheaper to build but used **about 21× more tokens per query**, so there is no break-even point. A decomposition-RAG variant closed most of the synthesis gap at lower token cost.
  - Judge agreement: Spearman ρ = 0.04 on holistic groundedness and 0.81 on the most concrete criterion.
  - **v3 correction:** v1/v2 truncated RAG evidence to 1,500 characters before citation scoring. After the fix, and after checking against 2 human annotators on 100 claims (scorer agreement 50–54%, then 65–69%), the wiki's citation advantage disappears.
  - Source: [arXiv](https://arxiv.org/abs/2605.18490).
- **[V] Cochran**, "Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation", arXiv 2607.04576 (Jul 6 / Oct 7, 2026).
  - Setup: a real 709-page LLM-maintained knowledge base, with access structures varied (catalog plus one-line summaries) while the pages were held identical.
  - In the pilot, a capable agent **never loaded the index** and went straight to pages it guessed.
  - Retrofitting cut cost in every condition, with quality non-inferior under the main protocol. Quality was not shown under forced catalog loading or stricter criteria, and human–judge agreement was poor.
  - Source: [arXiv](https://arxiv.org/abs/2607.04576).
- **[V] Ming, Li, Wu, Que**, "Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki", arXiv 2605.25480 (May 2026).
  - Documents are compiled into bidirectionally linked wiki pages. Search, read and follow-link are exposed as tools, and an "Error Book" handles self-correction.
  - It scores +2.0–8.1 F1 over HippoRAG 2, LightRAG and GraphRAG on HotpotQA, MuSiQue and 2WikiMultiHopQA, and is best on AuthTrace (multi-document structured queries).
  - The abstract does not compare against plain chunk RAG. **[S]** A snippet says agents read about 2.5–3.9 pages per query vs RAG's fixed 5 passages.
  - Source: [arXiv](https://arxiv.org/abs/2605.25480), [HTML v2](https://arxiv.org/html/2605.25480v2).
- **[V] Brian Wright**, "Potential for Enhanced Learning in Machine Learning Classes by Using Wiki LLM Indexing", arXiv 2609.25303 (Sep 21, 2026).
  - Setup: a single ML-course corpus and 59 questions, comparing the best vector RAG from the author's prior work against a Karpathy-style compiled wiki, scored by an LLM judge against a human rubric.
  - Single-fact: RAG 9.33 vs wiki 9.96. Questions that need linking: wiki 9.93 and 100% grounded vs RAG 8.14 and 64% grounded. The author says "further testing is needed".
  - Source: [arXiv](https://arxiv.org/abs/2609.25303).
- **[S] Open-source implementations** include nashsu/llm_wiki (desktop app) and the Open Knowledge "knowledge base starter pack" with provisional and canonical folders. There are also many blog explainers.
  - Sources: [DenserAI blog](https://denser.ai/blog/llm-wiki-karpathy-knowledge-base/), [Inkeep Open Knowledge docs](https://open-knowledge-docs.preview.inkeep.com/docs/workflows/karpathy-llm-wiki), [classmethod](https://dev.classmethod.jp/articles/karpathy-llm-knowledge-base/).
  - The gist's comment thread is dominated by people promoting their own tools ([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)).

### Inferences
- WiCER is the only paper in the cluster that studies the **small model with the whole wiki in context (KV-cache)** regime and **omission** as the primary failure. The Cochran, Ming and Wright papers all use agentic browsing with a strong model and study cost, synthesis or groundedness.
- Cochran's 21× query-token overhead applies to agentic browsing. WiCER's full-context/KV regime removes most of that per-query cost, which is a clear differentiator.
- The cluster's methodological lessons apply directly to WiCER's evaluation hygiene:
  - LLM judges disagree with each other and with humans (ρ = 0.04 on holistic criteria).
  - Scorer truncation bugs can flip conclusions.
  - Concrete criteria are far more reliable than holistic ones.
  - These echo the project's own pitfalls notes (compile noise ≥ effect sizes).
- Wright reports "single-fact" parity, while WiCER finds catastrophic loss under blind compilation. Likely reasons are corpus size relative to wiki budget, compiler model strength, and question granularity (course concepts vs RepLiQA's detail questions). That is worth stating explicitly rather than treating the results as contradictory.

### Gaps
- I found no peer-reviewed (conference) version of any LLM-wiki paper yet. All are arXiv preprints from 2026.
- I did not search exhaustively. There are likely more 2026 preprints (e.g. lint/consistency maintenance, multi-agent wiki upkeep) that a second "LLM-wiki" search on arXiv listing pages would surface.

## 3. Benchmarks for wiki/article generation and how coverage, faithfulness and citations are measured

### Takeaway
The benchmarks measure coverage with heading or entity recall and rubric LLM-judges against a reference article, and citation quality with NLI-style citation recall and precision. None measures QA-answerability of the generated article for a downstream reader. The closest analogue, used by WiCER and Cochran, is QA over the compiled artifact.

### Cited Findings
- **[V] FreshWiki** (STORM).
  - Coverage metrics are heading soft recall, heading entity recall, article entity recall and ROUGE-1/L.
  - The LLM rubric scores Interest, Organization, Relevance and Coverage on a 1–5 scale.
  - Citation recall and precision are judged by an NLI-like LLM (Mistral-7B).
  - Source: [STORM HTML](https://arxiv.org/html/2402.14207v2).
- **[V] WildSeek** (Co-STORM): real information-seeking goals, used for automatic evaluation of discourse and report quality ([arXiv](https://arxiv.org/abs/2408.15232)).
- **[S] WikiSeek** (WikiAutoGen): multimodal, with under-covered topics ([arXiv](https://arxiv.org/html/2503.19065v1)).
- **[S] WikiGenBench**, "Exploring Full-length Wikipedia Generation under Real-World Scenario", arXiv 2402.18264, COLING 2025: 1,320 entries on newly emerging events ([arXiv](https://arxiv.org/abs/2402.18264), [GitHub](https://github.com/zhzihao/WikiGenBench)).
- **[V] Wiki Live Challenge**, Wang, Xu, …, Mao, Zhang, arXiv 2602.01590 (Feb 2026).
  - It is a live benchmark with 100 recent Wikipedia Good Articles as references.
  - "Wiki Eval" combines **39 writing-quality criteria** derived from the GA criteria with factual-verifiability metrics.
  - It finds a significant gap between deep-research agents and expert GA articles. The abstract gives no numbers.
  - Source: [arXiv](https://arxiv.org/abs/2602.01590).
- **[S] DeepResearch Bench**: RACE for quality, FACT for citation accuracy and effective citations ([arXiv](https://arxiv.org/pdf/2506.11763)).
- **[S] Adjacent benchmarks:**
  - SurGE (arXiv 2508.15658, scientific survey generation) reports incomplete topic coverage and reference hallucination ([arXiv](https://arxiv.org/pdf/2508.15658)).
  - WETBench (arXiv 2507.03373) detects machine-generated Wikipedia edits ([arXiv](https://arxiv.org/html/2507.03373)).
- **[S] CodeWikiBench** (CodeWiki, below) has 21 repos with hierarchical rubrics and LLM-judge scoring. It explicitly rejects BLEU/ROUGE ([arXiv](https://arxiv.org/html/2510.24428v5)).
- **[V]** In the LLM-wiki-vs-RAG preprints, evaluation is downstream QA with LLM judges.
  - Cochran uses synthesis, groundedness and claim-level citation support, plus a human check ([2605.18490](https://arxiv.org/abs/2605.18490)).
  - WiCER uses RepLiQA QA scores and catastrophic-failure rate ([2605.07068](https://arxiv.org/abs/2605.07068)).

### Inferences
- For WiCER, the gap in the literature is a metric for **fact retention / omission rate** of a compiled artifact relative to its sources. That would be the inverse of citation precision. Atomic-fact recall (FActScore-style decomposition of the source, then a check of whether the wiki entails each fact) would bridge the article-generation benchmarks and the QA-based evaluation. I did not find a paper that does exactly this for compiled wikis.

### Gaps
- "WikiBench" as named in the brief: I did not find a benchmark by that exact name in this pass, and it may be confused with WikiGenBench or WikiSeek.
- I did not open RACE/FACT formulas or the WikiGenBench metrics.

## 4. Wikis written FOR LLMs (machine-readable / compressed pages, llms.txt, code wikis)

### Takeaway
LLM-facing documentation is emerging as practice: the llms.txt convention, DeepWiki and CodeWiki for code, and progressive-disclosure catalogs. Rigorous evaluation of *LLM-consumer* utility, as opposed to human-judged quality, is thin. CodeWiki is judged by rubrics, and Cochran's progressive-disclosure work is the main study of LLM-facing wiki structure.

### Cited Findings
- **[V] llms.txt**, proposed by Jeremy Howard on 2024-09-03 (spec last modified 2026-08-10).
  - It is a markdown file at `/llms.txt`: an H1 name (required), a blockquote summary, H2 link lists with notes, and an "Optional" section agents can skip to save context.
  - Linked pages should have clean `.md` versions. The stated aim is to avoid cluttered HTML and wasted tokens.
  - Source: [llmstxt.org](https://llmstxt.org/).
- **[S] CodeWiki**, "Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases", FSoft-AI4Code, arXiv 2510.24428 (v5 Jan 14, 2026).
  - Method: hierarchical decomposition, recursive multi-agent processing with dynamic delegation, and bottom-up synthesis with architecture and data-flow diagrams across 7 languages.
  - It scores 68.79% vs 64.06% for **DeepWiki** (Cognition's closed system), but trails on C/C++ (53.24 vs 56.39). CodeWikiBench covers 21 repos. Prior work cited: RepoAgent, Open-DeepWiki.
  - Source: [arXiv HTML](https://arxiv.org/html/2510.24428v5).
- **[V] Progressive disclosure** (Cochran 2607.04576): a compact catalog plus one-line summaries reduces agent cost on a 709-page LLM-maintained knowledge base without demonstrated quality loss. However, agents may ignore the index ([arXiv](https://arxiv.org/abs/2607.04576)).
- **[V] Karpathy's gist**: index.md (catalog with one-line summaries) and log.md (append-only) are the LLM-facing navigation layer ([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)).
- **[V] "Retrieval as Reasoning"**: the wiki is designed as an agent-native retrieval structure with links and tools ([arXiv](https://arxiv.org/abs/2605.25480)).

### Inferences
- All LLM-facing designs found assume a **tool-using agent** that navigates pages. WiCER's setting, where a whole compact wiki is placed in a small model's context or KV cache, is a different design point. Here page navigation aids matter less than **density and verbatim retention per token**. llms.txt's "Optional" section is a crude version of WiCER's budget/priority problem.
- DeepWiki/CodeWiki rubric scores (around 65–69%) suggest that even strong systems leave substantial content gaps in compiled documentation.

### Gaps
- I did not open DeepWiki's own documentation or any Cognition technical writeup.
- I found no paper that formally evaluates llms.txt's effect on QA accuracy.
- Context-compression methods (RECOMP, xRAG, gist tokens) are adjacent but were not researched here.

## 5. Known failure modes: hallucination, information loss, paraphrase drift, redundancy

### Takeaway
Article generators mainly fail by **adding** unsupported material (about 15% unsupported sentences, inferential over-linking, source bias). Compiled-wiki QA mainly fails by **omitting** facts: WiCER reports 53–60% catastrophic failures under blind compilation. Redundancy and shallow breadth motivate OmniThink. Evaluation itself is a failure mode, through judge disagreement and scorer bugs.

### Cited Findings
- **Hallucination / unsupported claims.**
  - STORM has about 15% of sentences unsupported by citations. Its top errors are improper inferential linking, inaccurate paraphrasing and irrelevant citations, plus "red herring" over-speculation and over-association of unrelated facts [V] ([STORM](https://arxiv.org/html/2402.14207v2)).
  - SurGE reports reference hallucination in survey generators [S] ([arXiv](https://arxiv.org/pdf/2508.15658)).
- **Paraphrase drift.** "Inaccurate paraphrasing" is an explicit STORM citation-error category [V] ([STORM](https://arxiv.org/html/2402.14207v2)).
- **Information loss.** WiCER's "compilation gap": blind compilation scores 2.14–2.32 vs 3.46 with 53–60% catastrophic failures, and targeted fact pinning recovers about 80% [V] ([WiCER](https://arxiv.org/abs/2605.07068)). STORM articles were judged less informative than revised Wikipedia [V] ([STORM](https://arxiv.org/html/2402.14207v2)). SurGE reports incomplete topic coverage [S].
- **Redundancy / shallowness.** OmniThink attributes repetitive, shallow output to vanilla RAG writing [V] ([OmniThink](https://arxiv.org/abs/2501.09751)).
- **Bias / tone.** In STORM's evaluation, 7/10 editors found articles "emotional" or "unneutral", inherited from web sources [V].
- **Scale degradation.** WiCER finds full-context inference falls below RAG at scale because of attention dilution [V]. Karpathy limits the index approach to about 100 sources and hundreds of pages [V].
- **Evaluation fragility.**
  - Cochran found LLM-judge ρ = 0.04 on holistic groundedness, scorer–human agreement of 50–69% on citation support, and that a truncation bug flipped the citation conclusion [V] ([2605.18490](https://arxiv.org/abs/2605.18490)).
  - STORM's human evaluation had low agreement, α 0.22–0.39 [V].
  - Wiki Live Challenge and CodeWikiBench rely on LLM rubric judges [V/S].
- **Agent behaviour mismatch.** Agents bypass the provided index and guess page locations [V] ([2607.04576](https://arxiv.org/abs/2607.04576)).

### Inferences
- A useful framing for WiCER's related-work section treats compilation errors as two-sided:
  - **commission** (hallucinated or over-linked claims), which the article-generation literature measures with citation precision;
  - **omission** (dropped details), which only WiCER measures directly.
- Verbatim-preserving compilation ("render" over paraphrase) targets both paraphrase drift and omission. The STORM error taxonomy is independent evidence that paraphrasing introduces errors.

### Gaps
- I found no study that quantifies paraphrase drift in compiled wikis specifically, for example numeric or date corruption rates.
- I found no study of redundancy's effect on reader-model QA under a fixed token budget.
- Multi-round wiki maintenance (Karpathy's ingest/lint loop) has no published longitudinal evaluation of drift or staleness that I could find.
