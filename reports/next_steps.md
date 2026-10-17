# Next Steps

## Readiness decision

The Stage 1 engineering smoke criteria are met: the unmodified I-JEPA objective and random mask path trained on real agricultural images, the model and EMA updated, and a checkpoint loaded and continued for another epoch. The loader, audit outputs, test suite, and isolated environment are in place.

**Do not begin full-scale Stage 2 pretraining on all 39,963 images yet.** Only 15,603 images currently have sufficiently confirmed source provenance and a usable source license record; 14,743 remain quarantined as `pending_audit`. The pHash/dHash scan is a conservative screening pass, but its 776 perceptual candidate edges have not all been visually adjudicated. The 9,606 PhenoBench rows are excluded, and split-crossing candidate edges were removed from the include-only manifests.

It is reasonable to begin Stage 2 method design or a bounded prototype using only the checked manifests, provided that the study scope explicitly uses this subset. Before a formal or broad pretraining run, complete the following gates:

1. Obtain or verify an authoritative mapping from the AgriField curator for the ten pending prefixes, including their original source IDs and source-level license/attribution terms. Update the registry and regenerate all audit products only from documented evidence.
2. Visually review perceptual candidate pairs/components, especially cross-split candidates, and record adjudications. Keep the current automatic screen as a reproducible first pass.
3. Confirm publication and redistribution obligations with the source terms, including CC BY-SA-derived materials and per-image iNat attribution. The compilation card is not by itself clearance for every component image.
4. Freeze a versioned manifest and data policy for the experiment; record the exact hashes in the run configuration/checkpoint.
5. Specify the Stage 2 agriculture-aware masking proposal and a controlled baseline/ablation plan before implementing that method. No such method is implemented in Stage 1.

The evidence is in `reports/data_audit.md`, `reports/data_audit.json`, `data/source_registry.json`, `data/exclusion_ledger.csv`, and `data/duplicate_pairs.csv`.
