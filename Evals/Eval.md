## Evaluation Plan

### 1. Human evaluation (primary)

- Pairwise preference (A vs B) with hidden ties allowed — most sensitive and fast.
- Dimension scores (1–5 or 1–7): faithfulness/consistency (no hallucinations), coverage/relevance, coherence/fluency, and overall quality/usefulness.
- Error annotation (lightweight MQM-style): mark spans for hallucination, omission of key content, and speaker leakage (task-specific for child-only summaries).
- Inter-rater reliability: report Cohen’s κ / Fleiss’ κ or Krippendorff’s α. Use ≥3 raters per item when possible.
- Tip: Randomize order, anonymize systems, pretrain raters with 3–5 gold examples, and compute paired significance (e.g., Wilcoxon signed-rank) on per-item differences.

### 2. Factual faithfulness (hallucinations)

- Use at least one faithfulness metric; they catch cases where text “sounds right” but isn’t supported by the source.
- Options: QAFactEval / SummaC / AlignScore / FactCC (pick one you can run easily).
- Report hallucination rate from human error tags as the ground truth, and show correlation with the automatic faithfulness score.

### 3. Semantic similarity / lexical overlap (for comparability)

Include a small set of widely used metrics so others can compare with you:

- ROUGE-1/2/Lsum (F1) — lexical overlap baseline.
- BERTScore-F1 (with IDF, rescaled) — semantic alignment baseline.
- (Optional) MoverScore or BLEURT/BARTScore — stronger semantic baselines if you have bandwidth.

### 4. Task-specific metrics (your project)

- Speaker purity / leakage: fraction of summary tokens attributable to the adult/interviewer (lower is better).
- Quote accuracy: exact/near-exact match rate of quoted child spans against child-speaker transcript.
- Compression ratio & redundancy: length vs. source child speech; duplicate n-gram rate.

### 5. What to report (clean, minimal set)

- Primary: Human pairwise preference and faithfulness error rate (hallucination %, leakage %).
- Secondary: SCU-coverage F1 (or QA-coverage), plus ROUGE-Lsum F1 and BERTScore-F1.
- Reliability: κ/α for human scores; bootstrap 95% CIs for all averages; paired tests for system deltas.
- Validation: Correlate each automatic metric with human overall/coverage and faithfulness to show which ones actually track human judgment in your setup.
1) Human evaluation (primary)

Pairwise preference (A vs B) with hidden ties allowed — most sensitive and fast.

Dimension scores (1–5 or 1–7): faithfulness/consistency (no hallucinations), coverage/relevance, coherence/fluency, and overall quality/usefulness.

Error annotation (lightweight MQM-style): mark spans for hallucination, omission of key content, and speaker leakage (task-specific for child-only summaries).

Inter-rater reliability: report Cohen’s κ / Fleiss’ κ or Krippendorff’s α. Use ≥3 raters per item when possible.

Tip: Randomize order, anonymize systems, pretrain raters with 3–5 gold examples, and compute paired significance (e.g., Wilcoxon signed-rank) on per-item differences.

Factual faithfulness (hallucinations)

Use at least one faithfulness metric; they catch cases where text “sounds right” but isn’t supported by the source:

QAFactEval / SummaC / AlignScore / FactCC (pick one you can run easily).

Report hallucination rate from human error tags as the ground truth, and show correlation with the automatic faithfulness score.

4) Semantic similarity / lexical overlap (for comparability)

Include a small set of widely used metrics so others can compare with you:

ROUGE-1/2/Lsum (F1) — lexical overlap baseline.

BERTScore-F1 (with IDF, rescaled) — semantic alignment baseline.

(Optional) MoverScore or BLEURT/BARTScore — stronger semantic baselines if you have bandwidth.

5) Task-specific metrics (your project)

Speaker purity / leakage: fraction of summary tokens attributable to the adult/interviewer (lower is better).

Quote accuracy: exact/near-exact match rate of quoted child spans against child-speaker transcript.

Compression ratio & redundancy: length vs. source child speech; duplicate n-gram rate.

What to report (clean, minimal set)

Primary: Human pairwise preference and faithfulness error rate (hallucination %, leakage %).

Secondary: SCU-coverage F1 (or QA-coverage), plus ROUGE-Lsum F1 and BERTScore-F1.

Reliability: κ/α for human scores; bootstrap 95% CIs for all averages; paired tests for system deltas.

Validation: Correlate each automatic metric with human overall/coverage and faithfulness to show which ones actually track human judgment in your setup.