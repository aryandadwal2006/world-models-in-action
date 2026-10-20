# Chapter 3 manuscript workspace

This folder is the section-by-section Markdown working copy of the uploaded Chapter 3 Word draft. It is based on the source text in `ch3_draft(1).docx`, checked against `chapter_03_design.md` and the current experiment-readiness findings. The original DOCX is not modified by this conversion.

## Reading order

1. [`00-introduction.md`](00-introduction.md)
2. [`3.1-learning-a-state-without-labels.md`](3.1-learning-a-state-without-labels.md)
3. [`3.2-pretext-tasks.md`](3.2-pretext-tasks.md)
4. [`3.3-contrastive-learning.md`](3.3-contrastive-learning.md)
5. [`3.4-collapse-and-anti-collapse.md`](3.4-collapse-and-anti-collapse.md)
6. [`3.5-masked-prediction.md`](3.5-masked-prediction.md)
7. [`3.6-latent-bottlenecks.md`](3.6-latent-bottlenecks.md)
8. [`3.7-temporal-prediction-cpc.md`](3.7-temporal-prediction-cpc.md)
9. [`3.8-assembling-the-encoder.md`](3.8-assembling-the-encoder.md)
10. [`exercises.md`](exercises.md)
11. [`summary.md`](summary.md)
12. [`appendix-a-proof-details.md`](appendix-a-proof-details.md)
13. [`figure-and-table-production.md`](figure-and-table-production.md)
14. [`references-and-credits.md`](references-and-credits.md)

## Source-of-truth rule

- Use `worldmodels/` for conceptual code listings; use `ch03/` for experiment configuration, figures, tables, and recorded results.
- Use the canonical result JSON files in `ch03/results/` for numeric claims. Do not overwrite canonical results while testing changes.
- Do not claim that CPC measures physical rollout, action-conditioned dynamics, or planning. Its primary result is held-out, in-batch future-latent retrieval; secondary results are linear state probes.
- The uploaded DOCX has no embedded images. It is a content/figure-direction draft, so this workspace preserves detailed production briefs without claiming the art is already present. The image filenames and detailed art briefs are listed in `figure-and-table-production.md`; they must be inserted or drawn separately during book production.
- Table 3.1 and Table 3.2 were missing from the DOCX and have been drafted here. Table 3.1 is placed before Table 3.2 in §3.1 so table numbering follows first appearance; its taxonomy provides an overview before the method-by-method treatment. Table 3.4 and Table 3.5 are each treated as one numbered table with panel A/B, avoiding duplicate numbering. The extra contract-check table is converted to an unnumbered checklist under §3.8.
- The VICReg code example is marked as supplementary and unnumbered so the numbered code-listing sequence stays aligned with the design (MAE Listing 3.10, bottleneck Listing 3.11, CPC Listing 3.12, encoder Listing 3.13).
- Appendix A receives the InfoNCE-bound proof and derivative algebra that the design document assigns to the appendix.

## Runtime and hardware

The prose now states that `worldmodels.train.get_device()` chooses CUDA automatically when `torch.cuda.is_available()` is true and CPU otherwise. It shows a code snippet to print the selected device and GPU name. This selection covers PyTorch model computation; MuJoCo/OpenGL rendering remains a separate dependency and cached data avoids rendering during subsequent training runs. Do not state or imply a benchmarked speed-up unless a hardware-specific runtime is recorded.

## Remaining author/production work

Review every Markdown section for voice, line-level polish, citations, formula rendering, and Manning pagination. The current layout is a content workspace, not a print-ready manuscript. In particular, Figure 3.7 must be regenerated with the corrected angle bins/sampling; Figure 3.9 needs its two diagnostic panels composed and the near-zero spread made visible; Figure 3.4 needs a batch-level relationship schematic; and Figures 3.1, 3.3, 3.5, 3.10, 3.11's conceptual panel, 3.13, 3.15, and 3.17 need final artwork. The detailed register specifies all instructions and captions.