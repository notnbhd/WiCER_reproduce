# Evaluating LLM-compiled knowledge artifacts (wikis/summaries) for information loss and faithfulness, and QA benchmarks for compile-then-read

Scope note: written for WiCER (arXiv 2605.07068): an LLM compiles a RepLiQA topic corpus (~17 topics) into a compact wiki that a smaller model reads in context. Verification labels:
- **[V]**: I opened the arXiv abstract (via the arXiv API) or the paper itself on 2026-11-07. Titles, authors and IDs come from arXiv metadata.
- **[S]**: secondhand. The claim comes from search-result summaries, a blog or press coverage, and I did not open the primary paper.
Abstention/unanswerable work (AbsenceBench, compression-aware abstention) is covered in the earlier review and is only touched here.

## Q1. Atomic-fact coverage and faithfulness metrics for long-form text and summaries, and their 2025–2026 successors

### Takeaway
Two families matter. **Precision/faithfulness** metrics (FActScore → SAFE → VeriScore → VeriFastScore; MiniCheck/AlignScore as cheap verifiers; VeriTrail for multi-step pipelines) check whether what the wiki says is supported by the source. **Recall/coverage** metrics (FineSurE/UniSumEval "keyfact completeness", SummHay "insight coverage", FABLES omission typology) check what the wiki dropped. WiCER's main failure is loss of details, so a keyfact-recall metric computed against the source corpus fits it best, run as a check on the wiki before any QA. 2025 meta-evaluation (Godbole & Jia) warns that all these automatic metrics are biased against heavily paraphrased outputs and against content drawn from far-away parts of the source. That bias applies directly to a paraphrasing compiler, so human validation on a sample is required.

### Cited Findings
**Decompose-then-verify (precision)**
- **FActScore**, Min, Krishna, Lyu, Lewis et al., EMNLP 2023, arXiv 2305.14251 **[V]**. Splits a generation into atomic facts and reports the percentage supported by a knowledge source. ChatGPT reaches only 58% on biographies. The automated estimator has under 2% error. `pip install factscore`. — [arXiv](https://arxiv.org/abs/2305.14251)
- **SAFE / LongFact**, Wei, Yang, Song, Lu et al. (Google DeepMind), NeurIPS 2024, arXiv 2403.18802 **[V]**. An LLM agent splits text into facts and verifies each one with Google Search. It agrees with crowd annotators 72% of the time and wins 76% of 100 disagreement cases. Proposes **F1@K**, which combines fact precision with recall measured against a target number of facts K. — [arXiv](https://arxiv.org/abs/2403.18802)
- **VeriScore**, Song, Kim, Iyyer, Findings EMNLP 2024, arXiv 2406.19276 **[V]**. Extracts only *verifiable* claims, which fixes FActScore/SAFE's assumption that every claim is verifiable. Factuality on one task does not predict factuality on another. — [arXiv](https://arxiv.org/abs/2406.19276)
- **VeriFastScore**, Rajendhran, Zadeh, Sarte, Li et al., Findings EMNLP 2025, arXiv 2505.16973 **[V]**. A fine-tuned Llama-3.1-8B extracts and verifies claims in one pass. It correlates with VeriScore at r=0.80 per example and r=0.94 per system, and runs 6.6× faster (9.9× without retrieval). — [arXiv](https://arxiv.org/abs/2505.16973)
- **Core**, Jiang, Zhang, Weir, Ebner et al., arXiv 2407.03572 (2024) **[V]**. Decompose-then-verify scores can be inflated by adding obvious or repeated subclaims. Core keeps only unique, informative subclaims. — [arXiv](https://arxiv.org/abs/2407.03572)

**Grounded fact-checkers (claim vs source document)**
- **MiniCheck + LLM-AggreFact**, Tang, Laban, Durrett, EMNLP 2024, arXiv 2404.10774 **[V]**. MiniCheck-FT5 (770M parameters) matches GPT-4 at grounding checks for about 400× lower cost. LLM-AggreFact combines grounding-check datasets into one benchmark. — [arXiv](https://arxiv.org/abs/2404.10774)
- Bespoke-MiniCheck-7B reports 77.4% balanced accuracy on LLM-AggreFact and is described as the top fact-checker at the leaderboard's launch. The license is CC BY-NC 4.0. I did not check whether newer models have since passed it **[S]**. — [Bespoke docs](https://docs.bespokelabs.ai/models/bespoke-minicheck); [leaderboard blog](https://llm-aggrefact.github.io/blog)
- **AlignScore**, Zha, Yang, Li, Hu, ACL 2023, arXiv 2305.16739 **[V]**. A 355M-parameter alignment function trained on 4.7M examples from 7 tasks. It matches or beats GPT-4-based metrics on 22 consistency datasets. — [arXiv](https://arxiv.org/abs/2305.16739)
- **QAFactEval**, Fabbri, Wu, Liu, Xiong, NAACL 2022, arXiv 2112.08542 **[V]**. A QA-based consistency metric, 14% better than earlier QA metrics on SummaC. QA-based and NLI-based signals complement each other. (Older, but it is the ancestor of "probe QA" evaluation, which WiCER uses.) — [arXiv](https://arxiv.org/abs/2112.08542)
- **LongDocFACTScore**, Bishop, Xie, Ananiadou, arXiv 2309.12455 (2023, LREC-COLING 2024) **[V]**. Extends consistency metrics to long sources by retrieving the relevant source sentences for each summary sentence. Ships the LongSciVerify annotations. — [arXiv](https://arxiv.org/abs/2309.12455)

**Multi-step pipelines (closest structural match to compile → read)**
- **VeriTrail**, Metropolitansky & Larson (Microsoft Research), ICLR 2026, arXiv 2505.21786 **[V]**. Detects closed-domain hallucination in processes with multiple generative steps, such as GraphRAG. It traces *where* unsupported content entered and how faithful content descends from the source through intermediate outputs. Releases the first datasets that contain all intermediate outputs plus human faithfulness labels. — [arXiv](https://arxiv.org/abs/2505.21786); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009378)

**Coverage / completeness (recall: what was lost)**
- **FineSurE**, Song, Su, Shalyminov, Cai et al., ACL 2024, arXiv 2407.00908 **[V]**. An LLM evaluator that scores faithfulness, **completeness** and conciseness. Completeness is the share of source *keyfacts* that align to summary sentences. It beats NLI-, QA- and LLM-based baselines, with the biggest gains on completeness and conciseness. Code: github.com/DISL-Lab/FineSurE-ACL24. — [arXiv](https://arxiv.org/abs/2407.00908)
- **UniSumEval**, Lee, Yun, Cai, Su et al., Findings EMNLP 2024, arXiv 2409.19898 **[V]**. Fine-grained human labels across domains and input lengths, including keyfact validation and alignment. Benchmarks 9 LLM summarizers and compares automatic evaluators. — [arXiv](https://arxiv.org/abs/2409.19898)
- **MSumBench**, Min, Lee, Ban, Deng et al., ACL 2025, arXiv 2506.00549 **[V]**. Multi-domain English/Chinese benchmark that scores faithfulness, completeness and conciseness. Finds **systematic bias when LLMs evaluate their own summaries**, which matters if the same model family compiles and judges. — [arXiv](https://arxiv.org/abs/2506.00549)
- **CREAM**, Gong et al., arXiv 2409.10883 (2024) **[V]**. Reference-free comparison of summaries using key-fact alignment plus Elo ranking. A practical fit for A/B comparisons of compiler variants. — [arXiv](https://arxiv.org/abs/2409.10883)
- **SummHay**, Laban, Fabbri, Xiong, Wu (Salesforce), EMNLP 2024, arXiv 2407.01370 **[V]**. Synthetic haystacks with known *insights*, so Coverage and Citation can be scored exactly. Long-context LLMs without a retriever score below 20% Joint, and even oracle-retrieval systems trail humans (56%) by more than 10 points. Also usable to study position bias. — [arXiv](https://arxiv.org/abs/2407.01370)
- **FABLES**, Kim, Chang, Karpinska, Garimella et al., COLM 2024, arXiv 2404.01261 **[V]**. 3,158 human-annotated claims from summaries of 26 books published in 2023–24. **No LLM rater correlates strongly with humans at detecting unfaithful claims.** Gives a typology of omission errors and finds summaries systematically over-emphasize events near the end of the book. — [arXiv](https://arxiv.org/abs/2404.01261)
- **BooookScore**, Chang, Lo, Goyal, Iyyer, ICLR 2024, arXiv 2310.00785 **[V]**. Studies coherence of chunked book summarization. **Incremental updating loses less detail than hierarchical merging** but is less coherent. Directly relevant to how a wiki compiler merges documents. — [arXiv](https://arxiv.org/abs/2310.00785)
- **TofuEval**, Tang, Shalyminov, Wong, Burnsky et al., NAACL 2024, arXiv 2402.13249 **[V]**. Binary LLM judges, GPT-4 included, did worse than specialized non-LLM metrics at detecting factual inconsistencies in topic-focused summaries. — [arXiv](https://arxiv.org/abs/2402.13249)
- **FACTS Grounding**, Jacovi, Wang, Alberti, Tao et al. (Google DeepMind), arXiv 2501.03200 (2025) **[V]**. Documents up to 32k tokens. Responses are first disqualified if they ignore the request, then judged for full grounding by *several* judge models to reduce the bias of any single judge. Uses public plus private splits. — [arXiv](https://arxiv.org/abs/2501.03200)

**Meta-evaluation caution**
- **Verify with Caution**, Godbole & Jia, arXiv 2501.14883 (2025) **[V]**. Re-evaluates 5 state-of-the-art factuality metrics on 11 datasets. The metrics disagree with each other, often misjudge system-level rankings, and are **biased against highly paraphrased outputs and against outputs drawing on far-away parts of the source**. — [arXiv](https://arxiv.org/abs/2501.14883)

### Inferences
- A **"keyfact recall against the source corpus"** metric (FineSurE-style completeness) is the most direct way to measure WiCER's information loss, and it needs no questions. Extract atomic facts from each RepLiQA document once, cache them, and for each compiled wiki label each fact *present-verbatim / present-paraphrased / altered / absent*. This gives a dense, low-variance outcome: hundreds of facts per topic against a few dozen questions. The *altered* class (wrong number, name or date) is the faithfulness failure. *Absent* is loss.
- Report wiki **precision** too: what share of wiki claims is supported by the corpus (MiniCheck or Bespoke-MiniCheck checking each claim against the document set). Retrieval is needed because the corpus is longer than the checker's window (LongDocFACTScore pattern).
- Because of Godbole & Jia and FABLES, the automatic labels must be checked by hand on a sample (for example 200 facts) before they are trusted. Paraphrase bias would systematically *understate* recall for WiCER's paraphrasing compiler.
- Do not let Haiku judge Haiku-compiled wikis without a cross-family check (MSumBench self-preference bias).
- VeriTrail's provenance framing fits the "compile noise" problem: it shows whether an error entered at compile time (fact absent or altered in the wiki) or at read time (fact present in the wiki, but the reader got it wrong).

### Gaps
- I did not find a 2025–2026 paper that defines a standard "lost information" metric for *corpus-to-wiki* compilation as such. The closest are FineSurE completeness, SummHay coverage and FABLES omissions.
- I could not confirm the current LLM-AggreFact leader, since I did not open the live leaderboard.
- Numbers for FActScore successors after VeriFastScore (2026) were not found in this pass.

## Q2. Benchmarks for QA over private/unseen corpora, and which suit compile-then-read

### Takeaway
RepLiQA is still the cleanest leakage-controlled choice for WiCER: fictional, human-written, never-online documents, with a gold answer paragraph, organized by topic. For comparability and external validity, add one benchmark of long *real* text published after model cutoffs (NovelQA, NoCha or DetectiveQA for book-level; Loong for multi-document "no document left behind"). Add one synthetic, controllable benchmark (SummHay for coverage, or NoLiMa-style low lexical overlap to test paraphrase robustness). RULER/BABILong/Michelangelo test the reader model's long-context ability rather than compilation, so they are optional.

### Cited Findings
- **RepLiQA**, Monteiro, Noël, Marcotte, Rajeswar et al. (ServiceNow), NeurIPS 2024 Datasets & Benchmarks, arXiv 2406.11811 **[V]**. Five test splits, four unreleased at publication and released over time. Each sample has (1) a human-written fictional reference document absent from the internet, (2) a question, (3) the answer, and (4) the **gold supporting paragraph**. Supports both QA and topic retrieval. — [arXiv](https://arxiv.org/abs/2406.11811); [HF](https://huggingface.co/datasets/ServiceNow/repliqa)
- **NovelQA**, Wang, Ning, Pan, Wu et al., ICLR 2025, arXiv 2403.12766 **[V]**. Manually annotated QA over English novels averaging more than 200K tokens. Models struggle most with **detail-oriented** and multi-hop questions. Gold answers are held behind a leaderboard (secondhand from the project page; not re-verified). — [arXiv](https://arxiv.org/abs/2403.12766)
- **NoCha**, Karpinska, Thai, Lo, Goyal et al., EMNLP 2024, arXiv 2406.16264 **[V]**. 1,001 minimally different true/false claim pairs about 67 recently published novels. No open-weight model beat chance and GPT-4o reached 55.8%. Pairs that need global reasoning are much harder than sentence-level ones. Paired-claim scoring cancels guessing. — [arXiv](https://arxiv.org/abs/2406.16264)
- **DetectiveQA**, Xu, Ye, Liu, Liu et al., arXiv 2409.02465 (2024) **[V]**. 1,200 questions over detective novels (>100k tokens) in Chinese and English, with reference reasoning steps and a step-wise metric. — [arXiv](https://arxiv.org/abs/2409.02465)
- **Loong**, Wang, Chen, Fu, Liao et al., EMNLP 2024, arXiv 2406.17419 **[V]**. Multi-document QA in which *every* document is needed for the answer. Task types are Spotlight Locating, Comparison, Clustering and Chain of Reasoning. RAG does poorly. The closest real-text match to "the wiki must keep every document". — [arXiv](https://arxiv.org/abs/2406.17419)
- **LongBench v2**, Bai, Tu, Zhang, Peng et al., ACL 2025, arXiv 2412.15204 **[V]**. 503 multiple-choice questions with 8k to 2M-word contexts across six categories. Human experts score 53.7% under 15 minutes, the best direct-answer model 50.1%, and o1-preview 57.7%. MCQ format avoids judge noise. — [arXiv](https://arxiv.org/abs/2412.15204)
- **HELMET**, Yen, Gao, Hou, Ding et al., ICLR 2025, arXiv 2410.02694 **[V]**. 7 application categories, controllable lengths up to 128K, and model-based metrics. NIAH does not predict downstream performance. Recommends the RAG tasks as a fast proxy. — [arXiv](https://arxiv.org/abs/2410.02694)
- **∞Bench**, Zhang, Chen, Hu, Xu et al., ACL 2024, arXiv 2402.13718 **[V]**. Average length above 100K tokens, synthetic and real tasks in English and Chinese. — [arXiv](https://arxiv.org/abs/2402.13718)
- **LooGLE**, Li, Wang, Zheng, Zhang, ACL 2024, arXiv 2311.04939 **[V]**. Documents from after 2022, over 24k tokens each, 6,000 generated plus more than 1,100 human QA pairs. Separates short-dependency from long-dependency questions. Retrieval helps short QA, while window extension helps little on long dependencies. — [arXiv](https://arxiv.org/abs/2311.04939)
- **FRAMES**, Krishna, Krishna, Mohananey, Schwarcz et al. (Google), NAACL 2025, arXiv 2409.12941 **[V]**. 824 multi-hop questions over Wikipedia (count not in the abstract; secondhand). Scores 0.40 with no retrieval and 0.66 with multi-step retrieval. Wikipedia-based, so **parametric leakage is high**; a poor fit for WiCER. — [arXiv](https://arxiv.org/abs/2409.12941)
- **BABILong**, Kuratov, Bulatov, Anokhin, Rodkin et al., NeurIPS 2024 D&B, arXiv 2406.10149 **[V]**. 20 reasoning tasks hidden in long distractor text. LLMs effectively use only 10–20% of their context. — [arXiv](https://arxiv.org/abs/2406.10149)
- **Michelangelo / Latent Structure Queries**, Vodrahalli, Ontañón, Tripuraneni, Xu et al. (Google DeepMind), arXiv 2409.12640 (2024) **[V]**. Synthetic, unleaked, automatically scored tasks that require "chiseling away" irrelevant context. — [arXiv](https://arxiv.org/abs/2409.12640)
- **RULER**, Hsieh, Sun, Kriman, Acharya et al. (NVIDIA), COLM 2024, arXiv 2404.06654 **[V]**. Only about half of models that claim 32K or more hold up at 32K. — [arXiv](https://arxiv.org/abs/2404.06654)
- **NoLiMa**, Modarressi, Deilamsalehy, Dernoncourt, Bui et al. (Adobe/LMU), ICML 2025, arXiv 2502.05167 **[V]**. Needles chosen to have minimal lexical overlap with the question. At 32K, 11 of 13 models fall below 50% of their short-context baseline, and GPT-4o drops from 99.3% to 69.7%. Shows that models lean on literal matching. — [arXiv](https://arxiv.org/abs/2502.05167)
- **Sufficient Context**, Joren, Zhang, Ferng, Juan et al. (Google), ICLR 2025, arXiv 2411.06037 **[V]**. Classifies each instance by whether the context is sufficient to answer. Large models answer wrongly instead of abstaining when it is not. Gives a ready-made autorater for "is the answer still in the compiled wiki?" — [arXiv](https://arxiv.org/abs/2411.06037)
- **Memory benchmarks that share the compile-then-read pattern**: LoCoMo (Maharana et al., arXiv 2402.17753) **[V]**; LongMemEval (Wu et al., ICLR 2025, arXiv 2410.10813; 500 questions over five abilities including abstention; 30% drop for long-context LLMs) **[V]**; MemoryAgentBench (Hu, Wang, McAuley, arXiv 2507.05257; turns long-context datasets into incremental multi-turn input) **[V]**. — [LoCoMo](https://arxiv.org/abs/2402.17753); [LongMemEval](https://arxiv.org/abs/2410.10813); [MemoryAgentBench](https://arxiv.org/abs/2507.05257)
- **Prior "compile to text" systems with published evaluations**: RECOMP (Xu, Shi, Choi, ICLR 2024, arXiv 2310.04408; compresses retrieved documents to 6% with little loss on QA) **[V]**; GraphRAG (Edge et al., arXiv 2404.16130; pre-generated community summaries judged on comprehensiveness and diversity by an LLM) **[V]**; STORM/FreshWiki (Shao et al., NAACL 2024, arXiv 2402.14207; Wikipedia-like articles, with expert editors flagging "source bias transfer and over-association of unrelated facts") **[V]**. — [RECOMP](https://arxiv.org/abs/2310.04408); [GraphRAG](https://arxiv.org/abs/2404.16130); [STORM](https://arxiv.org/abs/2402.14207)
- 2026 wiki generation: LLMpedia (arXiv 2603.24080) reports that the gap between benchmark and open-ended performance is about coverage more than hallucination. Wiki Live Challenge reports the best agent covering only about 30.8% of Wikipedia facts **[S]**, search summaries only. — [LLMpedia](https://arxiv.org/abs/2603.24080); [Wiki Live review](https://www.themoonlight.io/review/wiki-live-challenge-challenging-deep-research-agents-with-expert-level-wikipedia-articles)

### Inferences
- **Fit for compile-then-read, by structure:**
  - *Per-document corpus with gold evidence and a topic grouping*: RepLiQA (best), LooGLE (per document), Loong (per multi-document set).
  - *One long document = one "corpus"*: NovelQA, NoCha, DetectiveQA, ∞Bench En.QA/MC. Each book can be compiled into a wiki and then queried. NoCha's paired claims are the most noise-robust format.
  - *Synthetic with exact gold content units*: SummHay (insights repeat across documents, and the generator gives an exact keyfact list for coverage), Michelangelo, BABILong.
  - *Avoid for the main claims*: FRAMES and other Wikipedia-based sets, because parametric leakage confounds whether the wiki carried the answer.
- RepLiQA's gold paragraph allows **evidence-conditioned diagnosis**. For each question, check whether the gold paragraph's key fact survives in the wiki (Sufficient-Context autorater or keyfact match). This splits errors into compile loss and reader failure, and addresses "failures are detail-level, not doc-level".
- NoLiMa suggests a mechanism for the paraphrase problem. When the compiler rewrites wording, question–wiki lexical overlap drops, and small readers depend on literal matching. Recommended diagnostic: stratify accuracy by lexical overlap between question and wiki text, and between question and source text.
- Always report a **closed-book baseline** (reader with no wiki) and a **full-context/oracle baseline** (reader given the gold paragraph or full documents), so wiki scores sit between a floor and a ceiling. On RepLiQA the closed-book floor should be near zero, and any non-zero value measures leakage or guessing.

### Gaps
- I did not verify which of these benchmarks publish explicit per-topic splits besides RepLiQA, or the exact RepLiQA unanswerable rate. Project memory records about 20% unanswerable, measured internally.
- I found no published benchmark designed specifically for "compile corpus → small reader".

## Q3. Evidence on paraphrase/summary drift: lost numbers, names, dates; position effects

### Takeaway
Evidence for systematic detail loss is solid but scattered. LLM summaries overgeneralize (dropping qualifiers and scope), favor the start and end of inputs, lose detail under hierarchical merging, and drift toward abstraction as density rises. No single 2025–26 paper directly measures how often LLMs drop numbers, names and dates. NumPrec/NEPrec-style entity-overlap metrics are the cheap stand-in.

### Cited Findings
- **Generalization bias**, Peters & Chin-Yee, Royal Society Open Science 12:241776 (2025), arXiv 2504.00025 **[V]**. 4,900 summaries from 10 LLMs. Even when told to be accurate, DeepSeek, GPT-4o and LLaMA-3.3-70B overgeneralized in 26–73% of cases. LLM summaries were about 5× more likely than human ones to contain broad generalizations (OR 4.85, 95% CI 3.06–7.70). Newer models did worse. Press coverage says accuracy prompts made overgeneralization about twice as likely **[S]**. — [arXiv](https://arxiv.org/abs/2504.00025); [PsyPost](https://www.psypost.org/ai-chatbots-often-misrepresent-scientific-studies-and-newer-models-may-be-worse)
- **Lost in the Middle**, Liu, Lin, Hewitt, Paranjape et al., TACL 2024, arXiv 2307.03172 **[V]**. U-shaped accuracy over the position of relevant information. — [arXiv](https://arxiv.org/abs/2307.03172)
- **MiddleSum / context utilization in summarization**, Ravaut, Sun, Chen, Joty, ACL 2024, arXiv 2310.10570 **[V]**. First comprehensive study of position bias in LLM summarization (6 LLMs, 10 datasets, 5 metrics). Tests hierarchical and incremental summarization as fixes. — [arXiv](https://arxiv.org/abs/2310.10570)
- **FABLES** **[V]**: book summaries over-emphasize events near the end, and omissions follow a typology of key narrative elements. **BooookScore** **[V]**: hierarchical merging loses detail compared with incremental updating. — [FABLES](https://arxiv.org/abs/2404.01261); [BooookScore](https://arxiv.org/abs/2310.00785)
- **Chain of Density**, Adams, Fabbri, Ladhak, Lehman et al., NewSum 2023, arXiv 2309.04269 **[V]**. At fixed length, fitting in more entities makes summaries more abstractive and fused. There is an informativeness/readability trade-off, and humans preferred density close to human summaries. — [arXiv](https://arxiv.org/abs/2309.04269)
- **NumPrec / NEPrec**, Deroy, Ghosh, Ghosh, arXiv 2306.01248 (2023) **[V]** (metric details secondhand from a search snippet): the share of numbers and named entities in a summary that also appear in the source, used to flag hallucinated or altered entities in legal summaries. — [arXiv](https://arxiv.org/abs/2306.01248)
- **Paraphrase bias in metrics** (Godbole & Jia 2025, above) **[V]**: factuality metrics penalize heavily paraphrased content. A paraphrasing compiler is hit twice, losing detail and being measured badly. — [arXiv](https://arxiv.org/abs/2501.14883)
- **STORM** expert feedback **[V]**: "source bias transfer and over-association of unrelated facts" in generated wiki articles. That is a merging error a multi-document wiki compiler can make. — [arXiv](https://arxiv.org/abs/2402.14207)
- An EMNLP 2025 Industry paper on omission detection in medical summaries (EmbedKDECheck, aclanthology 2025.emnlp-industry.22) exists **[S]**; I did not open it. — [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.22.pdf)

### Inferences
- WiCER should measure drift **per fact type**. Tag source keyfacts as number, date, proper name, quantity-with-unit or other, then report survival by type (verbatim / paraphrased-correct / altered / dropped). This turns "the compiler paraphrases" from an anecdote into a table, and also checks the TACT result (verbatim rendering beats truncation by +0.56 pooled).
- Report position effects at two levels: where the fact sits in its source document, and where its document sits in the compile input order. Randomizing document order across compile seeds both measures and averages out position bias.

### Gaps
- No 2025–2026 primary study found that measures LLM summary loss of numbers, names and dates specifically, with rates.

## Q4. Statistical practice: compile variance, paired tests, sample size

### Takeaway
Treat each question as drawn from a super-population, **cluster by topic and document**, **pair** conditions on the same questions, and include **compile seed** as a random factor. With about 17 topics, CLT-based error bars are too narrow, so use a hierarchical bootstrap or a Bayesian model. Detecting differences of about 5 points needs several hundred paired questions *and* several independent compiles per arm, because otherwise compile noise dominates.

### Cited Findings
- **Adding Error Bars to Evals**, Evan Miller (Anthropic), arXiv 2411.00640 (2024) **[V]** (full text read):
  - Report standard errors with every score.
  - Use **clustered standard errors** when questions share a passage. They can be more than 3× the naive SE: DROP 1.34 vs 0.44.
  - Resample answers K times per question. In the worked example the variance factor is (1+2/K)/3.
  - Do not lower temperature to cut variance.
  - Use **paired differences**: SE_diff = √(SE_A² + SE_B² − 2·Corr·SE_A·SE_B).
  - Power formula: n = (z_{α/2}+z_β)² (ω² + σ_A²/K_A + σ_B²/K_B)/δ². Example: δ=3 points, 80% power gives n≈969, so new evals should have about 1,000 or more questions. With n=198, raising K from 1 to 10 lowers the minimum detectable effect from 13.2% to 7.5%.
  — [arXiv](https://arxiv.org/abs/2411.00640)
- **Don't use the CLT with fewer than a few hundred datapoints**, Bowyer, Aitchison, Ivanova, ICML 2025 position paper, arXiv 2503.01747 **[V]**. CLT intervals badly underestimate uncertainty on small benchmarks. Recommends Bayesian or exact frequentist alternatives; library at github.com/sambowyer/bayes_evals. — [arXiv](https://arxiv.org/abs/2503.01747)
- **Quantifying Variance in Evaluation Benchmarks**, Madaan, Singh, Schaeffer, Poulton et al. (Meta), arXiv 2406.10229 (2024) **[V]**. Measures seed variance. Continuous metrics have lower variance than discrete ones, and item-response-theory methods did not reduce variance much. — [arXiv](https://arxiv.org/abs/2406.10229)
- **Signal and Noise**, Heineman, Hofmann, Magnusson, Gu et al. (AI2), NeurIPS 2025, arXiv 2508.13144 **[V]**. A benchmark's signal-to-noise ratio predicts whether it supports reliable decisions. Switching to lower-noise metrics (e.g., perplexity instead of accuracy) and filtering noisy subtasks both help. The v1 abstract says 375 models; the NeurIPS page cites different counts **[S]**. — [arXiv](https://arxiv.org/abs/2508.13144)
- **With Little Power Comes Great Responsibility**, Card, Henderson, Khandelwal, Jia et al., EMNLP 2020, arXiv 2010.06595 **[V]**. Underpowered comparisons are common in NLP. Typical 2,000-sentence MT test sets have about 75% power to detect 1 BLEU. Provides power-analysis notebooks. — [arXiv](https://arxiv.org/abs/2010.06595)
- **Towards Reproducible LLM Evaluation**, Blackwell, Barry, Cohn, arXiv 2410.03492 (2024) **[V]**. Models are non-deterministic even at temperature 0 with fixed seeds. Proposes a cheap repeated-run method for prediction intervals. — [arXiv](https://arxiv.org/abs/2410.03492)
- FACTS Grounding averages several judge models to reduce single-judge bias **[V]**. — [arXiv](https://arxiv.org/abs/2501.03200)

### Inferences (worked numbers are my arithmetic, not from a source)
- **Paired binary power.** With two conditions scored on the same questions, the variance of the per-question difference is about p_d/n, where p_d is the discordant fraction (questions one arm gets right and the other wrong). If p_d ≈ 0.30:
  - n=400 gives SE ≈ 2.7 points and a minimum detectable effect ≈ 2.8×SE ≈ **7.7 points** (α=.05, 80% power).
  - n=1,000 gives an MDE ≈ **4.9 points**.
  - Without pairing, an n=400 comparison of arms near 50% has SE ≈ 3.5 points and MDE ≈ 10 points.
- **Compile noise as a random effect.** If compile-to-compile SD of topic accuracy is σ_c, the condition mean over C compiles has a compile variance term of σ_c²/C, and question count cannot reduce it. Project memory says compile noise is at least as large as the effects, so at least C=3, preferably 5, independent compiles per arm are needed (different seeds and shuffled document order). Report the between-compile SD as its own quantity.
- **Small-cluster correction.** With 17 topics, clustered SEs have about 16 degrees of freedom. Use a hierarchical bootstrap that resamples topics, then compiles, then questions, or use a Bayesian/GLMM logit model with random intercepts for topic, document, question and compile (Bowyer et al.). Report per-topic paired differences and a sign test over topics as a robustness check.
- **Lower-noise outcomes** (Signal and Noise; Madaan et al.): keyfact recall over hundreds of facts per topic and a graded or log-prob answer score have more signal per dollar than binary QA accuracy over about 20 questions per topic. Use QA as the headline and keyfact recall as the mechanism.
- **Reader resampling** (Miller, K samples per question) is cheap because reading is the cheaper step. It shrinks reader noise and leaves compile noise as the dominant term, which can then be measured cleanly.

### Gaps
- No published estimates of compile-to-compile variance for LLM-compiled knowledge artifacts were found. WiCER would be among the first to report it.

---

### Recommended evaluation protocol for WiCER (synthesis of the above)
1. **Splits.** Questions that guide or probe compilation must never be used for scoring. Report (a) probe accuracy, (b) truly held-out accuracy on questions never shown to the compiler and not generated from the same paragraphs, and (c) the gap between them as an overfitting measure. Stratify by evidence: "probe-covered paragraph" vs "uncovered paragraph", using RepLiQA's gold paragraph.
2. **Two-stage decomposition** (VeriTrail / Sufficient Context): for each held-out question, (i) is the gold fact present in the wiki, scored as verbatim / paraphrased / altered / absent by a keyfact matcher plus a MiniCheck-class checker; (ii) given presence, did the reader answer correctly? Report P(present) × P(correct | present) alongside end-to-end accuracy.
3. **Question-free coverage metric.** FineSurE-style keyfact recall over all source documents, broken down by fact type (number/date/name/other) and by position. Also report wiki precision (supported claims ÷ all claims). Validate both against about 200 hand-labelled facts, because of paraphrase bias (Godbole & Jia 2025).
4. **Baselines on the same questions.** Closed-book (leakage floor), truncation at matched token budget, verbatim-extract at matched budget, full-context or gold-paragraph oracle (ceiling), and a random-selection baseline at matched budget.
5. **Unanswerables.** Score the about 20% unanswerable questions separately: accuracy on answerable, abstention precision/recall on unanswerable. The earlier review covers the details.
6. **Statistics.** At least 3–5 compiles per arm, K≥3 reader samples, paired differences, hierarchical bootstrap (topic → compile → question) or a GLMM, 95% intervals, and a power statement such as "MDE ≈ X points at n, C". Use more than one judge, from a different model family than the compiler.
7. **External comparability.** Add one non-RepLiQA corpus that real text released after model cutoffs (NoCha paired claims, or Loong) plus SummHay for exact coverage, so results can be compared with published long-context and RAG numbers.
