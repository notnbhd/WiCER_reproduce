# KV-cache compression and quantization as an alternative to text-level context compression

Project framing: reader LLM in llama.cpp on a 24GB RTX 3090 must hold a compiled corpus; today we cut text to ~45% of source tokens. Option under consideration: q8_0 / q4_0 KV cache (`-ctk` / `-ctv`, V-quant requires flash attention `-fa`) to fit the full source instead. Our setting is query-agnostic by construction: the corpus is prefilled once and many questions are asked against it.

Annotated bibliography (each entry: title, first author et al., venue/year, link, finding, relevance). All arXiv IDs below were checked against arXiv or proceedings pages during this search.

## KV quantization: accuracy at 8/4/2-bit on long-context QA, and llama.cpp q8_0/q4_0 data

### Takeaway
Across independent benchmarks, KV quantization is the most robust KV-compression family: 4-bit (KIVI-4, FlexGen-4) is essentially lossless on LongBench and needle tests, 2-bit (KIVI-2) loses about 1 LongBench point, and 3-3.5-bit vector schemes (KVQuant, TurboQuant) are near-lossless. In llama.cpp specifically, q8_0 K+V costs ~0.005 PPL (negligible); the K cache is much more sensitive than V, so q8_0-K / q4_0-V is the safe 4x-ish point, while q4_0-K costs ~10x more KLD.

### Cited Findings

Bibliography:
- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** — Zirui Liu, Jiayi Yuan et al., ICML 2024 — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750). Finding: keys quantized per-channel, values per-token, recent window kept in full precision; 2.6x lower peak memory on Llama-2-7B with little/no accuracy drop, up to 4x batch and 2.35-3.47x throughput. Relevance: the per-channel-K insight explains why llama.cpp's per-token-block q4_0 on K hurts much more than on V.
- **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** — Coleman Hooper et al., NeurIPS 2024 — [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) / [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract.html). Finding: per-channel, pre-RoPE, non-uniform key quantization + 1% dense-and-sparse outliers gives <0.1 PPL degradation at 3-bit on Wikitext-2/C4 (LLaMA, LLaMA-2, Mistral); 1M context for LLaMA-7B on a single A100-80GB. Relevance: shows 3-bit is reachable without loss, but only with techniques llama.cpp's q4_0 does not use (pre-RoPE, per-channel keys).
- **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** — Amir Zandieh et al. (Google), 2025 (arXiv; venue not confirmed in this search) — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). Finding: data-oblivious (online, no calibration) vector quantization; "absolute quality neutrality" at 3.5 bits/channel and marginal degradation at 2.5 bits. A secondary write-up reports LongBench 50.06 at 3.5 bits (= full cache) and 49.44 at 2.5 bits for Llama-3.1-8B-Instruct ([Liner summary](https://liner.com/review/turboquant-online-vector-quantization-with-nearoptimal-distortion-rate); not checked against the paper's table). Relevance: online/calibration-free, so applicable to any prefilled corpus; community llama.cpp forks reportedly implement it (see Gaps).
- **KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches** — Jiayi Yuan et al., 2024 (alphaXiv BibTeX lists EMNLP 2024 Findings; arXiv page itself lists no venue) — [arXiv:2407.01527](https://arxiv.org/abs/2407.01527). Finding (Llama-3-8B-Instruct, LongBench avg / needle): baseline 45.2 / 100; KIVI-4bit (3.11x) 45.3 / 100; FlexGen-4bit (3.20x) 44.5 / 100; KIVI-2bit (5.05x) 44.3 / 100; with a harder 64-digit passkey, KIVI-2bit still 91.0% vs H2O 35.0% at 4x ([HTML v2](https://arxiv.org/html/2407.01527v2)). Relevance: the cleanest numeric case that 4-bit KV is free on long-context QA.
- **Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving** — Nikita Agrawal, Ruben Mayer, 2026 preprint — [arXiv:2607.05399](https://arxiv.org/abs/2607.05399). Finding: on Llama-3.1-8B and Mistral-7B LongBench-style tasks, KIVI-4 gives the most stable quality across models; SnapKV gives the best long-context throughput; "compression ratio alone is a poor predictor of end-to-end performance". Relevance: 2026 confirmation that 4-bit quantization is the safe default; no numeric values were visible on the abstract page.

llama.cpp-specific data:
- llama.cpp PR #7412 (research demo by JohannesGaessler, closed June 2024 as obsolete) measured PPL/KLD vs FP16 for KV types with FP16 weights (K/V): q8_0/q8_0 ΔPPL 0.0046, KLD 0.00098; f16-K/q4_0-V KLD 0.0050; q8_0-K/q4_0-V (6.5 bpv) ΔPPL 0.024, KLD 0.0051; q4_0-K/f16-V KLD 0.0329; q4_0/q4_0 (4.5 bpv) ΔPPL 0.209, KLD 0.0365. Author: "no significant quality loss from using q8_0 instead of FP16 for the KV cache"; K is much more sensitive than V; q4_0-V with FP16 elsewhere is more precise than q6_K weights with FP16 KV — [llama.cpp PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412). (Model not named on the page; short-context perplexity, not long-context QA.)
- Informal benchmark (Qwen2.5-Coder-7B, 8K, 12 prompts): q8_0 output stays close to f16; q4_0 shows a "cliff" in output similarity — [InventiveHQ](https://inventivehq.com/blog/kv-cache-quantization-quality-benchmark). Low-quality evidence (tiny sample, similarity metric).
- Community consensus on r/LocalLLaMA (Ollama KV-quant merge thread): q8_0/q8_0 fine, q8_0/q4_0 model-dependent, q4_0/q4_0 risky; some users report broken long-RAG output at q4 — [Reddit thread](https://lr.in.psf.lt/r/LocalLLaMA/comments/1h62u1p/ollama_has_merged_in_kv_cache_quantisation/m0cvl9g/?context=3). Anecdotal.
- Guide summarizing Ollama docs: q8_0 ~1/2 memory of f16 "usually no noticeable impact"; q4_0 ~1/4 with "small to medium loss" more noticeable at long context; V-cache quant requires flash attention — [llmhardware.io guide](https://llmhardware.io/guides/kv-cache-quantization-ollama-llama-cpp). Secondary.

### Inferences
- For us, `-ctk q8_0 -ctv q8_0` halves KV memory at essentially zero measured cost; `-ctk q8_0 -ctv q4_0` reaches ~6.5 bpv (~2.5x vs f16) at roughly 5x the KLD of q8_0/q8_0 but still well below a q4_0 K cache. q4_0 on K is the setting to avoid. Going from 45% of source to 100% of source needs ~2.2x more context; q8_0/q8_0 (~1.9x) almost covers that, q8_0/q4_0 (~2.5x) covers it.
- The literature's lossless 2-4-bit results (KIVI, KVQuant, TurboQuant) rely on per-channel / pre-RoPE / outlier handling that llama.cpp's block q4_0 lacks; do not transfer KIVI-2 numbers to llama.cpp q4_0.
- KV quantization is query-agnostic by nature (no token is dropped), which is exactly why it survives cache reuse (see next section). It is the lowest-risk alternative to our text cut.

### Gaps
- No primary-source long-context QA (RULER, LongBench, needle) numbers for llama.cpp q8_0/q4_0 KV were found; PR #7412 is short-context perplexity on an unnamed model. A 2026 blog claiming "no measurable cost to verbatim recall down to q4_0 at 140K tokens" (sanmai.github.io) returned 404 and could not be verified. We should measure on our own reader model and corpus (f16 vs q8_0/q8_0 vs q8_0/q4_0, held-out QA).
- TurboQuant's venue and exact LongBench table were not verified from the paper itself.

## Query-agnostic KV eviction/compression (compress once, reuse across queries) vs query-aware methods

### Takeaway
KVzip (NeurIPS 2025 oral) is the reference query-agnostic method: it scores KV pairs by how much the model needs them to reconstruct the context, keeps 30% of the cache with negligible loss across QA/retrieval/reasoning, and composes with 4-bit KV quantization. Other 2025 query-agnostic or query-free scorers (Compactor, KeyDiff, Q-Filters, Expected Attention, LagKV, KNorm) exist, several in NVIDIA kvpress, but none are in llama.cpp.

### Cited Findings
- **KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction** — Jang-Hyun Kim et al. (SNU / NAVER), NeurIPS 2025 (oral) — [arXiv:2505.23416](https://arxiv.org/abs/2505.23416) / [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/f4eaa4b8f2d08edb3f0af990d56134ea-Abstract-Conference.html). Finding: 3-4x smaller KV cache, ~2x lower FlashAttention decode latency, negligible loss on QA, retrieval, reasoning and code (LLaMA3.1-8B, Qwen2.5-7B/14B-1M, Gemma3-12B, up to 170K tokens); retrieval tasks hold at 30% retention, summarization tolerates ~10%. Composes with QServe W8A8KV4: 124K-token KV cache drops from 16.3 GB (16-bit) to 1.2 GB with 4-bit + 70% eviction, negligible degradation. Scoring costs ~2x prefill compute, <2% extra peak memory (≈66 s for 124K tokens on A100 at 2K chunk). A context-independent head-level variant costs <1 min once per model ([HTML v2](https://arxiv.org/html/2505.23416v2)). Relevance: exactly our use case (one corpus, many questions); combined with 4-bit quantization it implies >10x cache reduction, but requires an HF/kvpress stack, not llama.cpp.
- **Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores** — Vivek Chari, Benjamin Van Durme (JHU), 2025 preprint — [arXiv:2507.08143](https://arxiv.org/abs/2507.08143). Finding: training-free, query-agnostic token retention via randomized approximate leverage scores, plus a likelihood-only calibration that estimates how much a given context can be compressed; v2 reports full-cache LongBench parity with ~68% average KV reduction across 27 RULER/LongBench tasks (Qwen2.5, Llama 3.1); ships compactor-vllm. (v1 and v2 abstracts quote different ratios.) Relevance: the per-context calibration is directly useful for deciding how far a compiled corpus can be compressed.
- **KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments** — Junyoung Park et al. (Qualcomm AI Research), NeurIPS 2025 (per arXiv v4) — [arXiv:2504.15364](https://arxiv.org/abs/2504.15364). Finding: attention-free eviction by key diversity (evict keys most similar to an anchor), works block-by-block; on LongBench with an 8K budget (~23% KV reduction) the gap to full cache is <0.04% for Llama 3.1-8B / 3.2-3B. Relevance: query-free and chunked-prefill friendly (resembles llama.cpp's ubatch processing), but the savings at no loss are modest.
- **Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression** — Nathan Godey et al., 2025 preprint — [arXiv:2503.02812](https://arxiv.org/abs/2503.02812). Finding: scores KV pairs with a context-agnostic projection (SVD of calibration queries), FlashAttention-compatible; 99% needle-in-a-haystack at 32x compression, up to 65% smaller perplexity drop than StreamingLLM, competitive with SnapKV on retrieval; tested up to 70B. Relevance: query-independent at inference time (filters fixed per model), cheap; a plausible llama.cpp-portable scorer.
- **Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution** — Alessio Devoto, Maximilian Jeblick, Simon Jégou (NVIDIA), 2025 preprint — [arXiv:2510.00636](https://arxiv.org/abs/2510.00636). Finding: training-free; estimates how future, unseen queries will attend to each KV pair from activation statistics, so it does not need the real question; releases KVPress with 20+ methods. Third-party summary says it matches or beats SnapKV, TOVA, KeyDiff, KNorm, StreamingLLM, DuoAttention especially at high compression ([Moonlight review](https://www.themoonlight.io/review/expected-attention-kv-cache-compression-by-estimating-attention-from-future-queries-distribution); not checked in the paper). Relevance: designed for the "question not known at prefill" case.
- **NVIDIA kvpress** (library, not a paper) — [GitHub](https://github.com/NVIDIA/kvpress) / [HF blog](https://huggingface.co/blog/nvidia/kvpress). Finding: compression is applied only to context tokens "so that you can evaluate the compression for different questions", i.e. the benchmark protocol is query-agnostic by design; includes KVzipPress, FastKVzipPress, KVzapPress, KeyDiffPress, LagKVPress ("query free, attention-weight free, flash-attention compatible"), KnormPress, SnapKV, TOVA, Expected Attention, Q-Filters; supports stacking with transformers `QuantizedCache` (e.g. quanto 4-bit). RULER-4K leaderboard on Hugging Face. Relevance: the ready-made harness if we ever want to compare eviction + quantization on our corpus outside llama.cpp.

### Inferences
- Query-agnostic eviction at 30% retention (KVzip) plus 4-bit quantization would let a 3090 hold far more than the full source, but none of these methods run in llama.cpp; adopting them means moving the reader to HF transformers/vLLM. For simply fitting 100% of the source (~2.2x more than now), quantization alone looks sufficient and cheaper.
- Eviction is lossy at the token level, so it shares the risk profile of our text cut (detail-level loss, cf. our held-out displacement finding), whereas quantization is lossy only numerically.

### Gaps
- No head-to-head of KVzip vs Compactor vs Q-Filters under one protocol was located beyond each paper's own tables and the kvpress leaderboard (whose numbers were not retrieved).
- No evidence of any of these eviction methods being merged into llama.cpp was found.

## Evidence that query-aware eviction fails in multi-query / cache-reuse settings

### Takeaway
Yes, strongly. SCBench (ICLR 2025) shows SnapKV/PyramidKV/StreamingLLM collapse when one compressed cache serves multiple requests (e.g. Qwen2.5-72B SnapKV 1/32: 45.8 -> 3.3), and that SnapKV without the query scores 0.0 on first-query string retrieval; KVzip reports query-aware methods degrade even at 90% retention under multi-query reuse. KV quantization (KIVI) is the most robust compressor in the same setting.

### Cited Findings
- **SCBench: A KV Cache-Centric Analysis of Long-Context Methods** — Yucheng Li, Huiqiang Jiang et al. (Microsoft), ICLR 2025 — [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) / [MSR page](https://www.microsoft.com/en-us/research/?p=1112793). Finding: 12 tasks (string/semantic retrieval, global info, multi-task) in multi-turn and multi-request shared-context modes; "sub-O(n) memory methods suffer in multi-turn scenarios", KV compression methods "generally underperform in shared scenarios, offering only minor benefits in the first round". Avg scores (full cache multi-turn / multi-request): LLaMA-3.1-8B full 48.7 / 37.2, SnapKV 1/32 20.9 / 15.2, PyramidKV 1/32 21.0 / 15.3, StreamingLLM 1/32 15.5 / 14.5, KIVI 1/8 32.0 / 24.8, LLMLingua-2 1/3 28.2 / 23.1; Qwen2.5-72B full 52.9 / 45.8, SnapKV 26.5 / 3.3, StreamingLLM 18.9 / 2.2, KIVI 39.1 / 35.8. SnapKV without the query: 0.0 on first-query string retrieval and 9.7 on semantic retrieval (LLaMA-3.1-8B) ([HTML v2](https://arxiv.org/html/2412.10319v2)). Relevance: the closest benchmark to our setting (shared corpus, many queries); note budgets are aggressive (1/32 eviction, 1/8 quantization), not the 1/2-1/4 we need.
- **KVzip** (above) — query-aware methods "suffer from performance degradation even at a 90% cache budget ratio under multi-query scenarios"; Fig. 2 contrasts SnapKV with per-query prefill vs a reused cache on SQuAD — [arXiv:2505.23416](https://arxiv.org/abs/2505.23416).
- **The Pitfalls of KV Cache Compression** — Alex Chen, Renato Geh, Aditya Grover, Guy Van den Broeck, Daniel Israel, ACL 2026 (long) — [arXiv:2510.00231](https://arxiv.org/abs/2510.00231) / [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.acl-long.1926/). Finding: with StreamingLLM, SnapKV, TOVA, H2O, K-Norm on Llama3.1-8B and Qwen2.5-14B under multi-instruction prompts (IFEval), some instructions degrade much faster and are effectively ignored; instruction order and eviction bias drive system-prompt leakage; simple eviction-policy fixes help. Relevance: eviction removes content non-uniformly, an analogue of our "pinned vs unpinned docs" displacement.
- **Hold Onto That Thought: Assessing KV Cache Compression On Reasoning** — Minghui Liu et al., 2025 preprint — [arXiv:2512.12008](https://arxiv.org/abs/2512.12008). Finding: no single eviction strategy wins across tasks for Llama-3.1-8B-Instruct; for reasoning models H2O and a decoding-enabled SnapKV do best; low budgets can lengthen reasoning traces. Relevance: weak; decode-phase focus, no quantization comparison.

### Inferences
- Any eviction scored with the question (SnapKV, PyramidKV, H2O with query window) is the wrong tool for a corpus-once / many-questions reader; only query-agnostic scorers (KVzip, Compactor, Q-Filters, Expected Attention, KeyDiff) or quantization should be considered.

### Gaps
- SCBench does not report eviction at moderate budgets (1/2-1/4) in multi-request mode in the portion read; the failure magnitude at our needed ratio is not directly measured.

## Benchmarks comparing KV compression vs text (prompt) compression

### Takeaway
Two benchmarks put prompt compression (LLMLingua-2) side by side with KV methods: both find token-deleting prompt compression is the worst family for retrieval/needle tasks, while KV quantization is the best. SCBench notes LLMLingua-2 can help global/many-shot tasks.

### Cited Findings
- Yuan et al. 2024 ([arXiv:2407.01527](https://arxiv.org/abs/2407.01527)): "Keeping the prefill process uncompressed is crucial"; LLMLingua-2 on Llama-3-8B-Instruct: 2x -> LongBench 33.5 / needle 51.3; 4x -> 29.9 / 8.3; 8x -> 26.9 / 0.0, vs KIVI-2bit at 5.05x -> 44.3 / 100 and baseline 45.2 / 100 ([HTML v2](https://arxiv.org/html/2407.01527v2)).
- SCBench ([arXiv:2412.10319](https://arxiv.org/abs/2412.10319)): LLMLingua-2 at 1/3 improves global-information tasks such as many-shot ICL but "significantly" degrades retrieval; on LLaMA-3.1-8B it scores 28.2 / 23.1 (multi-turn / multi-request) vs KIVI 1/8 at 32.0 / 24.8 and full cache 48.7 / 37.2.
- QEvict (2026 preprint, [arXiv:2608.05326](https://arxiv.org/pdf/2608.05326)) notes eviction and quantization are evaluated under separate protocols because nominal settings do not give the same physical memory footprint; AnchorKV ([arXiv:2608.02901](https://arxiv.org/pdf/2608.02901)) compares SnapKV, PyramidKV, AdaKV, TurboQuant at matched byte budgets. (Seen only as search-result titles/snippets; not opened.)

### Inferences
- These benchmarks compare generic extractive prompt compression, not a compiled/curated text like ours, so they overstate the gap for our case. Still, the direction is consistent: at equal memory, keeping all tokens at lower precision beats deleting tokens, especially for detail/retrieval questions (which is where our held-out failures live).

### Gaps
- No benchmark found compares abstractive/LLM-compiled text compression (our approach) against KV quantization at matched memory.
- QEvict/AnchorKV/HqeKV (ACL Findings 2026, [anthology preview](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.201/)) were not opened; their numbers are unverified.

## Practical numbers: memory savings vs accuracy loss

### Takeaway
Rule of thumb from the sources: 8-bit KV (2x) ~free; 4-bit KV (~3-4x) ~free with good schemes, small-but-measurable loss with llama.cpp q4_0 on K; 2-2.5-bit (5-6x) costs ~1 LongBench point with KIVI/TurboQuant; query-agnostic eviction to 30% (3.3x) ~free with KVzip; eviction+4-bit stacks to ~13x on a 124K context.

### Cited Findings
| Setting | Memory vs fp16 | Quality | Source |
|---|---|---|---|
| llama.cpp q8_0/q8_0 | 8.5 bpv (~1.9x) | ΔPPL +0.005, KLD 0.001 | [PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412) |
| llama.cpp q8_0-K / q4_0-V | 6.5 bpv (~2.5x) | ΔPPL +0.024, KLD 0.005 | [PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412) |
| llama.cpp q4_0/q4_0 | 4.5 bpv (~3.6x) | ΔPPL +0.209, KLD 0.037 | [PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412) |
| KIVI-4bit | 3.11x | LongBench 45.3 vs 45.2, needle 100 | [arXiv:2407.01527](https://arxiv.org/html/2407.01527v2) |
| KIVI-2bit | 5.05x | LongBench 44.3 vs 45.2, needle 100 | [arXiv:2407.01527](https://arxiv.org/html/2407.01527v2) |
| KVQuant 3-bit | ~5x | <0.1 PPL | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| TurboQuant 3.5 / 2.5 bit | ~4.5x / ~6x | "quality neutral" / marginal | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| KVzip 30% retention | 3.3x | negligible loss (QA/retrieval) | [arXiv:2505.23416](https://arxiv.org/abs/2505.23416) |
| KVzip 30% + 4-bit KV | 16.3 GB -> 1.2 GB @124K | negligible | [arXiv:2505.23416](https://arxiv.org/html/2505.23416v2) |
| SnapKV 1/32, multi-request | 32x | Qwen2.5-72B 45.8 -> 3.3 | [arXiv:2412.10319](https://arxiv.org/html/2412.10319v2) |
| LLMLingua-2 4x (text) | 4x | LongBench 45.2 -> 29.9, needle 100 -> 8.3 | [arXiv:2407.01527](https://arxiv.org/html/2407.01527v2) |

### Inferences
- Fitting 100% of source instead of 45% requires ~2.2x headroom. q8_0/q8_0 gives ~1.9x; q8_0-K/q4_0-V gives ~2.5x and is the setting the evidence supports. Recommended experiment: same reader, same questions, (a) current 45% text + f16 KV, (b) full source + q8_0/q4_0 KV, (c) full source + q8_0/q8_0 if it fits; compare held-out accuracy (not just probes), with repeated runs given our compile-noise finding.
- Bigger reduction (KVzip-style eviction) is only worth it if we leave llama.cpp.

### Gaps
- Perplexity numbers (PR #7412) are short-context; how KV-quant error accumulates at 50-100K tokens in llama.cpp is not measured by any primary source found.
