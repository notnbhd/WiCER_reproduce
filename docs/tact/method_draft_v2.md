# 3 Method

> **Draft note (v2).** System name `TACT` (Typed, Anchored Compilation with conTracts) is a
> placeholder. Notation follows WiCER where possible so that the delta is legible to a reader who
> knows that paper.
>
> **Changes from v1.** Every change below is forced by a measurement on a pilot ledger
> (6 RepLiQA documents, 209 claims, 30 probes; plus a 80-probe hand-labelled taxonomy over one
> topic). Numbers are cited inline as *(pilot)* and reported in full in $\S$5.1.
> (i) The claim record gains a speaker slot and a relational claim type ($\S$3.2).
> (ii) Extraction gains entity canonicalization, and $\S$3.3 now states the failure mode that
> span-contiguity actually produces — silent fragmentation, not rejection.
> (iii) Type routing is rebuilt: multi-entity relation tables are replaced by entity-attribute
> blocks plus a per-document fact table, and narrative no longer carries claims ($\S$3.4.1).
> (iv) The contract is stated over *sets* and reports partial coverage ($\S$3.5.1).
> (v) The diagnosis taxonomy splits MISS-RELATION out of MISS-READ ($\S$3.5.2).
> (vi) Displacement is stated per probe, not per claim ($\S$3.7).
>
> **Changes from the WiCER reproduction (v2.1).** A 17-topic reproduction of WiCER (Haiku 5.5
> compiler and judge, Llama 3.1 8B reader, RepLiQA shortest-80) measured where its gain goes;
> numbers are cited as *(repro)* and traced in `ket_qua_repro_wicer.md`. The headline: WiCER's
> held-out change is +0.10 in documents whose probe was pinned and −0.13 in all others, while the
> compiler uses about a third of its budget. Every change below follows from that.
> (vii) Refinement is append-only into a reserved headroom; the loop never re-renders
> ($\S$3.5.1, $\S$3.5.3, $\S$3.7).
> (viii) The budget is allocated per document, and a headroom fraction is held back at
> iteration 0 ($\S$3.4.2).
> (ix) Probes are self-generated and dense; benchmark questions never enter the loop
> ($\S$3.2).
> (x) Pinning closes over a neighbourhood of $A(q)$, not $A(q)$ alone ($\S$3.5.1).
> (xi) Recurrent diagnoses are promoted to ledger-wide extraction passes ($\S$3.5.4).
> (xii) A short statement of what the method is evaluated against ($\S$3.9).

## 3.1 Overview

Blind knowledge compilation fails on two distinct axes that prior work conflates under the single
label "information loss." The first is a *rendering* failure: the compiler produces an artifact far
below the requested budget and discards specific facts to do so. The second is a *refinement*
failure: when a lost fact is re-injected as a preservation constraint, the compiler rewrites the
artifact freely and evicts unrelated facts to stay within budget — the random knowledge displacement
effect that causes WiCER's improvement to plateau after one or two iterations.

Both failures are measurable in a reproduction of WiCER, and both are larger than the probe-set
numbers suggest. *Rendering (repro):* asked for 75%, 50% and 25% of the corpus, the compiler
returns 14.6%, 14.6% and 14.1% — the target has no measurable effect on length — and at WiCER's
50% target the realized ratio ranges from 9.5% to 20.9% across 17 topics. *Refinement (repro):* on
held-out answerable questions, WiCER improves documents whose probe it pinned by +0.10 and degrades
every other document by −0.13; unpinned documents are three times as numerous, so the net change is
−0.075. The displacement is not forced by the budget, which is two-thirds unused. It happens because
each refinement step recompiles the whole artifact and nothing prevents it from rewriting content
the constraint never mentioned.

Both failures share a root cause: the compiled wiki is an opaque string, so neither the budget nor
the preservation constraint is mechanically checkable. Every guarantee is delegated to the
compiler's compliance with a prompt.

TACT removes that delegation. It introduces an intermediate representation — a **claim ledger** $L$
— that is extracted once from the corpus and persists across refinement iterations, and it renders
the wiki as **typed, schema-bearing content** whose every unit is mapped back to ledger entries by
a sidecar index. Because every unit of the wiki has a stable identity, the preservation constraint
becomes a set-membership assertion that code can verify and repair, and the marginal cost of
honoring it becomes bounded.

One design commitment distinguishes v2 and is worth stating up front, since much of $\S$3.4 follows
from it: **no fact is permitted to live outside a typed block.** Prose survives in the artifact only
as lead-in summary that carries no claim absent from the sidecar. A method that admits a narrative
residual admits a fragment on which its own contract does not hold, and — measured rather than
assumed — that fragment is not small (*pilot:* 36–57% of claims under the v1 routing rule,
depending on its threshold).

The pipeline has three stages (Figure X):

1. **Extraction** ($\S$3.3): $D \rightarrow L$, once, offline.
2. **Typed rendering** ($\S$3.4): $(L, B, P) \rightarrow (W, \sigma)$ under a code-enforced token
   budget $B$ and a must-render set $P$, emitting a sidecar map $\sigma$.
3. **Contract-enforced refinement** ($\S$3.5): verify, evaluate, diagnose, and grow $P$.

Stage 1 runs once and is extended lazily ($\S$3.5.4). Stage 2 also runs once, at full budget
less a reserved headroom; stage 3 is the iterative loop, and it extends the rendering by appending
rows rather than rendering again ($\S$3.5.1).

## 3.2 Problem Setup and Notation

Let $D = \{d_1, \dots, d_N\}$ be a topic's document collection and $B$ a token budget with
$B = r \cdot |D|$ for a target compression ratio $r$. Let $Q_{\text{probe}}$ be a diagnostic probe
set, $d_q$ the document that generated probe $q$, and $s(q, W) \in \{1, \dots, 5\}$ an LLM-judge
score. Following WiCER we treat $s = 1$ as a catastrophic failure and write
$F_t = \{q : s(q, W_t) \le 1\}$.

**Probe set composition.** RepLiQA questions include a designed *unanswerable* class, and its share
is large and positionally skewed (*pilot:* 22.5% of questions in one topic; 0% at question slot 1
rising to 61% at slot 5). An unanswerable probe cannot diagnose fact loss: no pinning action can
change its score, and its failures measure refusal calibration rather than the compilation gap. We
therefore construct $Q_{\text{probe}}$ from answerable questions only and report unanswerable
behaviour as a separate ablation ($\S$5). Prior work that samples one question per document without
this filter allocates roughly a fifth of its diagnostic signal to a quantity its loop cannot act on.

**Probe source.** Benchmark probes are also too sparse for the loss they are meant to find. Under a
blind compile, failure on held-out questions is detail-level and close to uniform across documents
(*repro:* every held-out question fails in 3.8% of documents, against 2.7% expected if failures were
independent), and 14.1% of held-out answerable questions score $\ge 4$ with the full corpus and
$\le 1$ with the wiki. One probe per document samples that loss at one point per document; a loop
that repairs only what its probes see cannot reach the rest.

TACT therefore generates its own probes. For each $d$, a question-writer LLM emits $m$ questions,
each grounded in a span of $d$ that it reports as $\texttt{verbatim}$ and that is validated exactly
as claim spans are ($\S$3.3). Grounding in source spans rather than ledger claims is deliberate: a
probe written from $L$ can never expose MISS-EXTRACT. The answer set is then computed mechanically,
$A(q) = \{c \in L(d) : \texttt{span}(c) \cap \texttt{span}(q) \neq \emptyset\}$, which gives the
procedure for $A(q)$ that $\S$3.5 depends on, at the cost of being a recall proxy: a multi-span
relational answer is mapped only through the spans the question-writer reports. Self-generated
probes are answerable by construction. Benchmark questions are used for evaluation only
($\S$3.9), so no evaluation question influences the artifact.

A **claim** is a record

$$c = \langle \texttt{id}, \texttt{doc}, \texttt{span}, (e, r, v), \tau, \texttt{src}, \texttt{verbatim} \rangle$$

where $e$ is a *canonical* entity identifier ($\S$3.3), $r$ a relation, $\tau$ an optional
timestamp, $\texttt{doc}$ and $\texttt{span}$ locate the claim in the source, and
$\texttt{verbatim}$ is the minimal supporting text. Two fields are new relative to v1:

- $\texttt{src}$, an optional **speaker**. Corpora of this genre are quote-dense, and attribution is
  frequently part of what the probe asks for (*pilot:* 12.5% of probes carry an attribution
  requirement, concentrated in the hardest class). "AR eliminates misunderstandings" and "according
  to Quinn, AR eliminates misunderstandings" are different claims and are judged differently;
  without $\texttt{src}$ the distinction is either lost or smuggled into $e$, where it corrupts the
  merge key.
- $v$, the **value**, which is either a literal (as in v1, where it was written $a$) or a
  *reference* $\texttt{id}'$ to another claim. The reference form gives the ledger a way to
  represent relations between propositions — cause, purpose-to-outcome, mechanism, consequence.
  $\S$3.3 explains why v1 could not represent them at all, and $\S$3.5.2 why their absence was
  invisible to the diagnosis.

We write $L(d)$ for the claims extracted from $d$ and $L = \bigcup_d L(d)$.

A **rendering** is a pair $(W, \sigma)$ where $W$ is the wiki served to the reader model and
$\sigma$ maps addressable units of $W$ (timeline entries, attribute rows, fact-table rows) to claim
identifiers. We write $\mathrm{cl}(W) = \mathrm{range}(\sigma) \subseteq L$ for the set of
*realized* claims. $\mathrm{cl}(W)$ is computed by lookup, not by inference: this is the property
the rest of the method depends on, and $\S$3.4.1 is designed so that it holds over the whole
artifact rather than over a fragment of it.

## 3.3 Claim Ledger Construction

**Extraction.** For each $d \in D$ we prompt an extractor LLM to emit claims in the schema of
$\S$3.2. Extraction is *exhaustive rather than salient*: the extractor is instructed to enumerate
every assertion that could ground a factual question, since a claim absent from $L$ can never be
pinned later ($\S$3.5.4 mitigates the residual gap).

**Entity canonicalization.** Extractors invent entity strings freely, and the resulting near-
duplicates are pervasive: *pilot:* 39% of extracted entities participate in at least one
containment pair — `Joe Harris` / `Joe Harris company` / `Joe Harris work`, or one head term with
eleven modifier variants. This is not cosmetic. Both the merge rule below and the block structure
of $\S$3.4.1 are keyed on $e$, so an uncanonicalized ledger silently defeats both: a verifier
querying $(e, r)$ finds nothing because the fact was filed under a sibling name.

We therefore canonicalize before merging. Extraction runs in two passes over each document: the
first emits an entity inventory, the second extracts claims with $e$ constrained to that closed
inventory. Containment and embedding-similarity clustering over the inventory resolves the residual
variants. Canonicalization is a *precondition* for $\S$3.3's merge rule and $\S$3.4.1's routing,
not an optimization; we report inventory size and post-merge collision rate in $\S$5.

**Span validation.** LLM-reported character offsets are unreliable. We discard the reported offsets
and recover them by fuzzy-matching $\texttt{verbatim}$ against $d$, rejecting any claim whose best
match falls below a similarity threshold.

It is important to state what this filter does and does not catch, because the measurement is
counter-intuitive. *Pilot:* the rejection rate is **zero** — 209 of 209 claims matched exactly. This
is not evidence that the filter works; it is evidence that it never fires. An extractor required to
supply contiguous supporting text does not emit a claim that fails validation. It **decomposes**:
faced with a proposition whose support spans two non-adjacent sentences, it emits two individually
valid claims and drops the relation that joined them.

Concretely, for a probe whose gold answer is *"saved money on manuals while reducing paper waste,
resulting in financial savings and a more eco-friendly approach"*, the extractor emits a claim for
the saving, a claim for the waste reduction, and nothing for the *resulting in*. Both claims pass
validation, enter $L$, are rendered, and satisfy the contract. The answer the probe asks for is not
recoverable from either.

The reference-valued claims of $\S$3.2 exist to close this gap: a relational claim
$\langle e, \texttt{results\_in}, \texttt{id}' \rangle$ carries a $\texttt{verbatim}$ covering only
the connective and its immediate context, and is validated against that. Where no contiguous
connective exists, the relational claim is admitted with a *multi-span* provenance record — an
ordered list of spans rather than one — and is flagged as such, since it is the one claim class
whose provenance guarantee is weaker than the rest. *Pilot:* 54% of causal chains in the probe
taxonomy require the multi-span form.

**Provenance-aware merging.** Prior structured-generation work deduplicates extracted events and
triples by embedding similarity against thresholds. That rule is appropriate when many retrieved web
sources describe one event, but it is unsafe here: our documents are mutually independent, and
superficially similar assertions across documents routinely carry different values. Applying a
similarity-based deletion inside a system whose objective is fact preservation is self-defeating.

We replace it with a rule keyed on provenance and value rather than on surface form. For claims
$c_i, c_j$ agreeing on $(e, \tau)$ — with $e$ canonical, per the paragraph above:

| Case | Action |
| --- | --- |
| $\texttt{doc}_i \ne \texttt{doc}_j$, values agree | merge; retain both source references |
| $\texttt{doc}_i \ne \texttt{doc}_j$, values disagree | **retain both**; emit a cross-document contradiction record |
| $\texttt{doc}_i = \texttt{doc}_j$, values agree | merge (paraphrase duplicate) |

No claim is deleted on similarity grounds alone. The middle row recovers, as a byproduct, the class
of cross-page contradictions that prior wiki-construction work detects only through sampling-based
consistency checks.

## 3.4 Typed Rendering under an Enforced Budget

### 3.4.1 Type routing

v1 derived content type from relation frequency: relation signatures recurring across at least $m$
entities induced a table whose columns were relations and whose rows were entities; everything else
fell to narrative. Measured on real extractions, that rule fails on both of its objectives.

*Coverage.* The relation distribution has a long tail — *pilot:* 209 claims produced 106 distinct
relations, and 57% of entities carry exactly one relation. No threshold is satisfactory: $m = 2$
admits two-row tables that are tables in name only, while $m = 3$ leaves 46% of claims in narrative.
Read strictly as the frequent-itemset computation v1 describes, $m = 3$ admits two relation pairs
across six documents, covering 15% of claims.

*Density.* The rule is also more expensive than the prose it replaces. A multi-entity table over a
long-tailed relation set is overwhelmingly empty — *pilot:* 1008 cells, 88 filled, 9% density — and
empty cells still cost delimiter tokens. Measured against verbatim prose as the unit baseline, v1
routing costs **1.68×**. The information-density argument for typed rendering is sound where the
entity × relation grid is dense; on independent single-document narratives it inverts.

v2 replaces the rule with one keyed on entities rather than relations:

- **Timeline (T2).** Claims with non-null $\tau$, grouped by entity thread, sorted chronologically.
- **Entity block (T3a).** Each entity carrying $\ge 2$ distinct relations receives its own
  attribute block: rows are relations, cells are values. One block per entity.
- **Fact table (T3b).** All remaining claims — those whose entity appears exactly once — are
  collected into a single per-document table with columns $(e, r, v)$.
- **Lead summary (T1).** Prose survives only as an optional lead-in above a block. It is generated
  *from* the block it introduces and is constrained to assert nothing absent from $\sigma$. It
  carries no claim and contributes nothing to $\mathrm{cl}(W)$.

The threshold $n = 2$ in T3a is not a tuned parameter: $n = 1$ degenerates to putting every entity
in T3b, which is the same content under a different header, so the rule has no sensitive knob. The
routing is exhaustive by construction — every claim lands in T2, T3a, or T3b — so
$\mathrm{cl}(W) = L$ up to budget truncation, and the keyed-lookup property of $\S$3.6 holds over
the entire artifact.

Cost, measured: *pilot:* **0.95×** the prose baseline, against 1.68× for v1 routing. The saving is
structural rather than incidental — the layout has no empty cells, because every row exists because
a claim exists. Under a fixed $B$ equal to the prose cost of the full ledger, v2 routing fits 105%
of the ledger where v1 routing fits 59%.

### 3.4.2 Budget enforcement

Compilers reliably ignore stated word-count targets, over-compressing by factors of two to three and
thereby discarding specific facts long before the budget requires it. Instructing more firmly does
not fix this. TACT enforces the budget in code: rendering proceeds block by block, the realized
token count is measured after each block, and rendering continues until the count enters
$[(1 - \epsilon)B, B]$. Because the ledger supplies a ranked queue of unrealized claims, "continue"
is a well-defined operation — append the next rows — rather than a request for the model to expand
prose.

The size of this effect is the largest single quantity in the reproduction. A compiler that
realizes ~15% of the corpus against a 50% target (*repro*, $\S$3.1) serves the reader roughly a
third of the context it was allotted, and the full-corpus reader scores 4.00 against the blind
wiki's 2.96 on held-out answerable questions. Enforcement is the only component of TACT that acts
on every question rather than on the probed ones.

Two allocation rules sit on top of enforcement:

- **Per-document shares.** Each document receives $B_d = B \cdot |d| / |D|$, and unused share is
  redistributed only after every document is filled. A single global ranked queue would let
  claim-dense documents starve sparse ones. We note that the reproduction does *not* show this
  failure in the blind compiler — its losses are spread evenly ($\S$3.2) — so the rule guards
  TACT's own queue rather than correcting an observed WiCER defect.
- **Headroom.** Iteration 0 fills only to $(1 - \rho)B$, and the remaining $\rho B$ is reserved
  for pinned rows ($\S$3.5.1). Without it, the first repair either overflows $B$ or evicts, and
  eviction is displacement.

This is a precondition for the rest of the method. We note, against v1, that typed rendering
increases the number of facts that fit in $B$ tokens only under the routing of $\S$3.4.1; the
claim is a property of the layout, not of typing as such, and $\S$5 reports it as a measurement
rather than asserting it.

### 3.4.3 Sidecar provenance

Provenance can be carried in-context (per-row anchors, per-block source lists) or out-of-context (a
sidecar file). Under a fixed budget, in-context provenance is paid for in evicted facts: at roughly
four to six tokens per row, per-row anchors consume budget that would otherwise hold additional rows.

TACT stores $\sigma$ as a sidecar index mapping `(page, block, row)` to claim identifiers. The wiki
served to the reader contains no identifiers; audit and verification operate on the sidecar. This
differs from prior wiki systems that embed source sections in the page, and the difference is a
consequence of the setting: those systems serve an agent that traverses a file tree, whereas the
compiled wiki here is loaded whole into a bounded context.

We treat the in-context/out-of-context choice as an empirical question and ablate it ($\S$5), since
visible provenance may improve the reader's grounding on unanswerable questions even as it costs
coverage.

## 3.5 Contract-Enforced Refinement

### 3.5.1 The contract

Let $P_t$ be the set of pinned claim identifiers at iteration $t$. TACT maintains the invariant

$$\mathcal{I}_t : \quad P_t \subseteq \mathrm{cl}(W_t).$$

After each rendering, a verifier computes $P_t \setminus \mathrm{cl}(W_t)$ by set difference. Any
missing claim is repaired by *local insertion* — appending the corresponding row to its typed block
— not by recompilation.

Two clarifications relative to v1, both forced by the observation that a probe is answered by a set
of claims rather than by one. *Pilot:* $|A(q)| = 2.36$ on average, and 80% of answerable probes
require $|A(q)| \ge 2$.

- **The contract is over sets.** Pinning probe $q$ means pinning all of $A(q)$, and the verifier
  reports *partial coverage* $|A(q) \cap \mathrm{cl}(W_t)| / |A(q)|$ rather than a boolean. A probe
  whose answer set is half realized is a distinct state from one that is absent, and the loop
  should be able to see it.
- **Repair is local, but not single-touch.** Restoring $A(q)$ inserts $|A(q)|$ rows, and where
  $A(q)$ spans multiple entities it touches multiple blocks (*pilot:* 28% of probes). Repair
  therefore costs $O(|P_t \setminus \mathrm{cl}(W_t)|)$ rows across at most that many blocks, and
  touches no content outside them — which is weaker than v1's phrasing and is what the mechanism
  actually guarantees.
- **Repair is append-only.** $W_{t+1} = W_t \oplus \mathrm{rows}(P_{t+1} \setminus
  \mathrm{cl}(W_t))$: every block not receiving a row is byte-identical across iterations. Rows are
  drawn from the headroom $\rho B$ ($\S$3.4.2). When the headroom is exhausted, the evicted rows
  are the lowest-ranked unpinned rows of $d_q$ first and of other documents only after that, so
  displacement lands first on the document that benefits from the pin. This is the direct remedy for
  the reproduction's central measurement: under recompilation, unpinned documents lose 0.13 on
  held-out questions (*repro*, $\S$3.1); under append-only repair with headroom remaining, they
  lose nothing by construction.
- **Pins close over a neighbourhood.** WiCER's pins help the *other* questions of a pinned document
  (*repro:* +0.10 on held-out questions in pinned documents), presumably because the recompile
  gives that document's section more attention. Append-only repair forgoes that side effect, so
  TACT makes it explicit: pinning $q$ pins $N(A(q))$, the claims of $d_q$ that share an entity with
  $A(q)$ or whose span falls in the same paragraph. This is a hypothesis about where the spillover
  came from, not a measured mechanism, and it costs budget: per-probe pin size becomes
  $\bar{k}_N = \mathbb{E}|N(A(q))| \ge \bar{k}$ ($\S$3.7).

Prior work states an analogous monotonicity property but grounds it in the assumption that the
compiler includes all pinned facts before allocating budget elsewhere, while simultaneously
acknowledging that the compiler is stochastic and that no guarantee prevents a pinning constraint
from having side effects. $\mathcal{I}_t$ replaces that assumption with a postcondition that holds
by construction of the repair step.

### 3.5.2 Diagnosis taxonomy

For a failed probe $q \in F_t$, let $A(q) \subseteq L(d_q)$ be the claims that answer it. The ledger
permits a four-way attribution that is unavailable to text-level diagnosis. The fourth category is
new in v2 and is the reason the taxonomy was revisited:

- **MISS-EXTRACT:** $A(q) \not\subseteq L(d_q)$ — the fact never entered the ledger. An extraction
  failure, not a compression failure.
- **MISS-RENDER:** $A(q) \subseteq L(d_q)$ but $A(q) \cap \mathrm{cl}(W_t) = \emptyset$ — the
  compression step dropped it. The failure mode prior work implicitly assumes is the only one.
- **MISS-RELATION:** every element of $A(q)$ is realized, but the probe asks for a relation *between*
  them that is itself absent from $L$. The constituent claims are present and correct; the
  connective is not.
- **MISS-READ:** $A(q) \subseteq \mathrm{cl}(W_t)$, the relation is representable and present, and
  the reader still failed.

MISS-RELATION was invisible in v1, and its invisibility was structural rather than accidental. Its
constituent claims are in $L$, so MISS-EXTRACT does not fire; they are in $\mathrm{cl}(W)$, so
MISS-RENDER does not fire; the probe therefore fell into MISS-READ — the one branch the loop
records but never repairs. A system with v1's taxonomy reads a class of *extraction* loss as a
*reader* deficiency, and concludes that its structured rendering is hard to read when in fact its
ledger cannot express what the probe asks for.

The distinction matters because the two have opposite remedies. MISS-RELATION is repairable:
targeted extraction of a reference-valued claim ($\S$3.2), pinned like any other. MISS-READ is not
repairable by any preservation strategy and bounds what the method can achieve.

Only MISS-RENDER and MISS-RELATION enter $P_{t+1}$. MISS-EXTRACT triggers ledger extension
($\S$3.5.4); MISS-READ is reported as a separate diagnostic.

*Pilot:* under the probe taxonomy of $\S$5.1, 16% of probes ask for a relation between propositions.
This is the population at risk of being misattributed to MISS-READ under v1, and it is the quantity
the split is designed to expose.

### 3.5.3 Loop

```
Input:  D, budget B, headroom rho, probes per doc m, max iterations T
Output: W*

L          <- Extract(D)                          # 3.3, once, with canonicalization
Q_probe    <- GenerateProbes(D, m)                # 3.2, span-grounded, A(q) by span overlap
(W, sigma) <- Render(L, B, rho)                   # 3.4, once; fills to (1-rho)B, per-doc shares
P          <- {}
for t = 0 .. T-1:
    scores <- Judge(Q_probe, W)
    F      <- {q : scores[q] <= 1}
    if F = {} or improvement < delta: break
    new    <- {}
    for q in F:
        case Diagnose(q, L, cl(W)):               # 3.5.2
            MISS-RENDER  : new <- new U N(A(q))
            MISS-RELATION: L <- L U ExtractRelation(d_q, q); new <- new U N(A(q)) U {rel}
            MISS-EXTRACT : L <- L U ExtractTargeted(d_q, q); new <- new U N(A(q))
            MISS-READ    : record only
    L      <- L U Promote(diagnoses)              # 3.5.4, ledger-wide pass for recurrent classes
    P      <- P U new
    W      <- AppendRows(W, P \ cl(W), rho)       # 3.5.1, append-only; invariant I_t holds after
return best W
```

Two differences from the WiCER loop are worth flagging. First, the artifact is rendered once and
then only extended: no iteration re-renders from $L$, let alone recompiles from $D$, so the
expensive corpus pass is paid once. Second, refinement cannot touch a block it does not append to,
so while headroom remains, displacement is not merely bounded but zero ($\S$3.7).

### 3.5.4 Lazy ledger extension

Extraction recall bounds what can ever be pinned, which is a regression relative to text-level
diagnosis: reading $d_q$ at failure time gives unrestricted access to the source. We recover the
lost capability by seeding $L$ with bulk extraction and extending it on demand: a MISS-EXTRACT or
MISS-RELATION diagnosis triggers a targeted extraction pass over $d_q$ conditioned on $q$, and the
resulting claims are appended to $L$ with full provenance. The ledger grows along the failure
trajectory, so extraction effort concentrates where it is demonstrably needed.

The two triggers differ in what they request. MISS-EXTRACT asks for assertions the bulk pass missed.
MISS-RELATION asks specifically for a reference-valued claim joining ledger entries that are already
present, which is a narrower and more reliable request than open-ended re-extraction.

**Promotion.** A targeted pass repairs one document, but the reproduction shows that the loss a
probe exposes is a sample from a uniform population ($\S$3.2), so the same kind of loss almost
certainly sits in documents no failing probe touched. When a diagnosis recurs across at least $k$
probes with the same signature — MISS-EXTRACT of one claim class (dates, quantities,
attributions), or MISS-RELATION of one connective type — the targeted request is promoted to a
pass over all of $D$. Promotion is how a probe failure can improve held-out questions: a pin repairs
one answer, while a promoted pass changes what the ledger contains everywhere.

## 3.6 Why Structure and Contracts Are Interdependent

The two components of TACT are usually presentable as separate contributions. They are not separable
here, in either direction.

**Contracts require structure.** Verifying $c \in \mathrm{cl}(W)$ over free prose requires an
entailment check, i.e. a second model call whose output is approximate — reintroducing exactly the
approximate verification that motivated the contract. Over a typed block, the same check is a keyed
lookup: does a row keyed $(e, r)$ exist, and does it hold value $v$? Structure is what makes the
contract decidable.

The claim is stated without qualification only because of the routing commitment in $\S$3.4.1. A
routing rule that leaves a narrative residual leaves a fragment on which the contract silently
degrades to entailment, and the size of that fragment is a property of the corpus rather than of the
method — *pilot:* 36–57% of claims under v1 routing. Under v2 routing the residual is empty by
construction, and the claim is literal rather than approximate. We regard this as the main reason to
prefer entity-keyed routing, independent of its token cost.

**Structure requires contracts.** Typed rendering imports a deduplication step, and
similarity-based deduplication deletes facts — the precise failure the system exists to prevent.
Provenance-keyed merging ($\S$3.3) is only definable once claims carry document identity and a
canonical entity key. Contracts are what make structure safe.

**What the commitment costs.** Forbidding fact-bearing prose converts a verification problem into a
readability problem. An artifact of attribute blocks and a fact table reads like a database dump,
and the reader must perform joins the prose used to perform for it. We consider this the correct
trade: the verification problem is undecidable within the method and its damage is unobservable,
whereas the readability problem surfaces as a MISS-READ rate that $\S$5 measures directly and that
lead summaries ($\S$3.4.1) are designed to mitigate. The trade is only defensible if that rate is
measured, which is why MISS-READ is reported as a headline number rather than a footnote.

## 3.7 Displacement Analysis

Let $\bar{c}$ be the mean token cost of a realized claim and $\kappa$ the marginal token cost of
forcing a pinned claim into the rendering. Under a fixed budget $B$, the number of unpinned claims
that fit is

$$n^{\text{free}}_t = \frac{B - \kappa |P_t|}{\bar{c}},$$

so the expected displacement between consecutive iterations is

$$\Delta_t \;=\; n^{\text{free}}_t - n^{\text{free}}_{t+1} \;=\; \frac{\kappa}{\bar{c}} \left( |P_{t+1}| - |P_t| \right).$$

Displacement is thus linear in the cost ratio $\kappa / \bar{c}$, and the two renderings differ only
in that ratio:

- **Prose refinement.** Honoring a preservation constraint requires the compiler to reconstruct
  surrounding narrative — connectives, referents, framing — so $\kappa_{\text{prose}} \gg \bar{c}$.
  Each pinned fact evicts several unpinned ones.
- **Typed refinement.** A pinned claim is realized as one additional row in an existing block, so
  $\kappa_{\text{typed}} \approx \bar{c}$ and $\Delta_t \approx |P_{t+1}| - |P_t|$: pinning one
  claim displaces at most one claim.

**Headroom.** With a reserve of $\rho B$ ($\S$3.4.2) and append-only repair ($\S$3.5.1), pinned
rows consume the reserve before they consume anything else, so

$$n^{\text{free}}_t = \frac{B - \max(\rho B,\ \kappa |P_t|)}{\bar{c}},$$

and $\Delta_t = 0$ for every iteration with $\kappa |P_{t+1}| \le \rho B$. Beyond that point the
linear regime above applies, with eviction ordered to fall on $d_q$ first. The reproduction gives
the quantity this should be compared against: under recompilation, unpinned documents lose 0.13
on held-out answerable questions (*repro*). The prediction is that under TACT the same split shows
no loss on unpinned documents while headroom lasts, and it is falsifiable at the noise floor of
$\S$3.9.

**Units.** The identity above is stated per *claim*, but $P_t$ grows by probe: a failed probe
contributes $A(q)$, not a single claim. Per probe the displacement is therefore
$\bar{k} \cdot (\text{probes newly pinned})$, with $\bar{k} = \mathbb{E}|A(q)|$ — *pilot:*
$\bar{k} = 2.36$ measured on a real ledger, against 1.38 estimated by counting propositions in gold
answers without extracting them. Since $\S$5 measures displacement per probe, the per-claim
statement understates the effect by that factor, and we report both. With neighbourhood closure
($\S$3.5.1) the per-probe factor is $\bar{k}_N$ rather than $\bar{k}$, which spends headroom faster;
$\rho$ and the closure rule trade against each other and are reported together.

This is an accounting argument, not a bound on answer quality; it predicts the *mechanism* we
measure in $\S$5 (displacement rate), and the prediction is falsifiable — if the measured
displacement rate does not fall in proportion to the measured $\kappa/\bar{c}$, the explanation is
wrong even if aggregate quality improves. It also assumes a fixed $\bar{c}$ across iterations, which
is only approximately true, since pinned rows skew toward high-value, long-value claims.

## 3.8 Instrumentation

Because $\mathrm{cl}(W)$ is computable, TACT exposes metrics that require no judge:

- **Claim coverage** $|\mathrm{cl}(W_t)| / |L|$ — a deterministic proxy for compilation quality,
  usable to tune the rendering stage offline before spending judge calls.
- **Pin compliance** $|P_t \cap \mathrm{cl}(W_{t+1})| / |P_t|$ measured *before* repair — the degree
  to which an unassisted compiler honors preservation constraints. Prior work cannot measure this
  quantity and assumes it is one.
- **Partial answer coverage** $\mathbb{E}_q |A(q) \cap \mathrm{cl}(W_t)| / |A(q)|$ — the set-valued
  refinement of pin compliance ($\S$3.5.1), and the quantity that predicts probe success better
  than claim coverage does.
- **Realized compression ratio**, measured rather than requested.
- **Contradiction count**, from the merge rule of $\S$3.3.
- **Entity collision rate** after canonicalization ($\S$3.3) — the precondition on which both the
  merge rule and the block structure depend.
- **Routing mix** — the fraction of claims in T2 / T3a / T3b, and the realized token cost per
  claim in each. Reported per topic, since the density result of $\S$3.4.1 is corpus-dependent.
- **MISS-READ rate**, reported as a headline number for the reason given in $\S$3.6.
- **Headroom remaining** $\rho B - \kappa|P_t|$ per iteration — the point at which displacement
  is predicted to start ($\S$3.7).
- **Held-out change split by pinned and unpinned documents** — displacement observed in score
  space rather than predicted in token space. This is the split that exposes WiCER's −0.13
  (*repro*), and it is the test of $\S$3.7. It needs the judge but no extra compiles.

We report these alongside judge-based quality throughout $\S$5.

## 3.9 What the Method Is Evaluated Against

The reproduction fixes four commitments, because each one changes which conclusion the numbers
support.

- **Held-out questions decide.** The headline comparison is on benchmark questions that never
  entered the loop, answerable and unanswerable reported separately ($\S$3.2). Because TACT's
  probes are self-generated, all benchmark questions are held out for TACT; for comparability every
  arm is scored on the same set, namely the benchmark questions WiCER did *not* use as probes.
- **Effects are judged against the compile noise floor.** Two compiles of an identical prompt
  differ by 0.26 and 0.40 on held-out answerable questions in the two topics where it was
  measured (*repro*). Per-topic differences below that are not reported as effects; claims are made
  on the 17-topic mean.
- **Iteration 0 is an arm of its own.** Budget enforcement acts on every question, and the loop
  acts on probed ones, so most of a held-out gain may come before the loop runs. TACT without
  refinement is reported separately from TACT, so that the gain is attributed to the right
  component.
- **Two baselines beyond WiCER.** (a) *Truncation:* each document's text cut to its share
  $B_d$, with no model in the loop — the ceiling on what budget filling alone buys. (b) *WiCER with
  self-generated probes* — so that a TACT advantage cannot be explained by probe density alone.

---

## Open drafting questions (v2)

1. **Where the pilot lives.** $\S$3 now cites measured quantities inline to justify design choices
   that v1 asserted. Either they stay as forward references to $\S$5.1, or the pilot becomes a short
   $\S$4 ("Design measurements") that precedes the main evaluation. The second reads better but
   makes the paper look like it has two evaluations.
2. **Corpus dependence of $\S$3.4.1.** The density inversion is measured on one topic of one
   benchmark, whose documents are independent single-author narratives. A corpus with a dense
   entity × relation grid would favour v1 routing. The honest framing is that routing should be
   *selected per corpus by measurement*, which weakens the contribution to "we provide the
   measurement"; the aggressive framing is that the dense-grid case is rare in compilation settings.
   Undecided.
3. **MISS-READ as a headline number.** $\S$3.6 now stakes the central design trade on it, which
   invites "your structured wiki is harder to read than prose" as the paper's own summary of itself.
   The mitigation (lead summaries) is stated but unmeasured. If the ablation shows lead summaries
   recover most of the gap, this becomes a strength; if not, $\S$3.6 needs a hedge.
4. **Baseline naming.** $\S$3.5.1 and $\S$3.7 describe WiCER by mechanism rather than by name to
   keep the contrast from reading as an attack. v2 adds several places where v1-of-this-paper is
   contrasted with v2, which is a different rhetorical problem — a method section that argues with
   its own earlier draft. Probably these should become plain statements with the measurement
   attached, and the "v1 said X" framing should survive only in the change note.
5. **Multi-span provenance** ($\S$3.3) is the one claim class whose provenance guarantee is weaker
   than the rest, and the paper currently flags it without saying what an auditor should do with the
   flag. Either define an audit procedure or restrict reference-valued claims to the contiguous case
   and accept the loss.
6. **Who renders.** Append-only repair ($\S$3.5.1) and code-enforced filling ($\S$3.4.2) make the
   typed rows a deterministic function of $(L, P, B, \rho)$; an LLM is left only the T1 lead
   summaries. That makes $\mathcal{I}_t$ hold trivially, and makes pin compliance ($\S$3.8) a
   measurement of the *baseline's* compiler rather than of TACT. The draft should say so
   plainly instead of implying an LLM renderer. It also leaves the T1 constraint ("asserts nothing
   absent from $\sigma$") as the one place where verification is still entailment.
7. **Attribution.** If the iteration-0 arm of $\S$3.9 carries most of the held-out gain — which
   the reproduction's 2.96 → 4.00 gap and the budget measurement make likely — the paper's
   held-out contribution is budget-enforced extractive rendering, and the contract loop is a
   probe-set and displacement contribution. The framing should be fixed before the numbers arrive,
   not after.
8. **Reproduction-specific numbers.** Every *(repro)* figure comes from one reproduction with a
   Haiku 5.5 compiler, one compile per arm, whose blind baseline sits ~0.6 above the WiCER paper's.
   The mechanism (pinned +0.10, unpinned −0.13) should transfer; the magnitudes may not. The noise
   floor is measured on 2 of 17 topics.
