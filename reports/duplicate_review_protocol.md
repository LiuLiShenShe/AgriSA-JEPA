# Duplicate review protocol

This is the protocol a human follows to adjudicate the perceptual duplicate candidates emitted by Stage 1. It exists because a perceptual hash is a screening signal, not proof.

## How the candidates were generated

`tools/audit_data.py` computed three things for all 39,963 images:

- SHA-256 of the raw file bytes.
- SHA-256 of the fully decoded RGB pixel buffer.
- A 64-bit pHash and a 64-bit dHash, both computed from the decoded RGB image using a DCT-II basis for pHash.

An edge is emitted in either of two ways:

1. **`kind=sha256`** — identical file bytes or identical decoded pixels. These are duplicates by construction and are recorded as `exact_duplicate` by this protocol. 8 edges.
2. **`kind=perceptual_candidate`** — pHash Hamming distance <= 4 **and** dHash Hamming distance <= 6. The intersection of two independent screeners is used because field scenes are visually uniform and a single hash produces collisions between unrelated images. This is a screening rule with false positives. 776 edges.

Every edge is then embedded in a connected component, and Stage 1 kept one deterministic representative per component (preferring the validation split). A component touching any PhenoBench row was excluded in full.

Stage 2B re-emits all 784 edges into `data/duplicate_review.csv`, enriched with split, source, path and manifest-membership context, and generates the offline contact sheet in `reports/duplicate_review/`.

## What "duplicate" means here

A pair is a **duplicate** if the two images depict the same physical scene at effectively the same moment, such that a model can memorise one from the other. Practically:

- the same field patch from the same camera position, differing only by compression, resizing, a small exposure or white-balance shift, or a small crop offset; or
- two frames of the same video or robot sequence that are close enough in time that the visible plants have not materially moved; or
- a re-encoded or re-cropped copy of the same original image.

A pair is **not a duplicate** if the two images are different captures of the same field at different times, different positions, different seasons, or different growth stages. Repetition of subject matter is expected in agricultural data and is not leakage.

The borderline case is consecutive frames in a sequence where plants have moved slightly. The rule is: if a human can tell the two images apart at a glance and both carry usable visual signal, they are **not** duplicates, even if they are adjacent in time. Flag them as `near_duplicate` instead so the decision is visible.

## Decision categories

| Verdict | Meaning | Effect |
| --- | --- | --- |
| `exact_duplicate` | Identical bytes or identical decoded pixels. | Treated as certain duplication. No visual inspection needed. |
| `confirmed_duplicate` | Human judged the pair to be the same scene. | Real leakage. |
| `near_duplicate` | Same scene, minor temporal or photometric offset, both usable. | Recorded, but not treated as leakage by the conservative rule. |
| `not_duplicate` | Human judged the pair to be different scenes. | The edge is retired. |
| `pending_manual_review` | Initial value. Nobody has looked. | Conservative rule applies. |

## Review order

1. **Cross-split candidates (247 edges).** One endpoint in train, the other in val. If confirmed, these are direct pretraining-to-validation leakage. These are section A of the contact sheet.
2. **Manifest-affecting candidates (11 edges).** One endpoint is in an include manifest and the other is not. Section B. A confirmed duplicate retires the excluded endpoint permanently; a `not_duplicate` verdict shows the Stage 1 component rule dropped a row that did not need dropping.
3. **Same-split screening candidates (526 edges).** Neither endpoint is in a manifest, so the frozen manifests are unaffected either way. Section C. Lowest priority, still worth sampling to calibrate the screener.

Within each section the contact sheet is sorted by `kind` (exact first), then by ascending pHash distance, then by ascending dHash distance, then by `edge_id`, so the most suspicious pairs come first.

## How a verdict is folded back

The reviewer works in `data/duplicate_review.csv`, one row per edge. `review_status` starts at `pending_manual_review` and is overwritten with the verdict above. `reviewer` holds the reviewer identifier, `review_notes` holds free text, `reviewed_at` holds an ISO-8601 date. `edge_id` is the stable key: it is `edge-<n>` where `n` is the 1-based row index in `data/duplicate_pairs.csv`, so it never changes as long as Stage 1 is not re-run. The contact sheet keys off the same `edge_id`, and its "Collect verdicts" button emits JSON keyed by it, so the sheet and the CSV can be reconciled row by row.

A verdict is only accepted if all four of `review_status`, `reviewer`, `review_notes` and `reviewed_at` are non-empty. `reviewer` is deliberately left blank by this tool — an unattributed verdict is not a verdict.

If a confirmed or near-duplicate verdict lands on an edge whose endpoints are in the frozen manifests, the freeze is invalidated: bump `data_version`, regenerate `data/frozen/`, and record the change in `reports/data_freeze.md`. Do not edit a frozen manifest in place.

## Conservative rule applied to unreviewed candidates

Until an edge carries an accepted human verdict, it is treated as if it *might* be a duplicate, and the freeze isolates it. The rule is:

> An edge triggers isolation if **any** of: (a) `cross_split` is true, (b) exactly one endpoint is in an include manifest, or (c) both endpoints are in include manifests and land in different ones.

For each triggered edge, **both** endpoints are excluded from the frozen manifests, whichever split they are in and whichever manifest they are in. A candidate edge is a two-sided warning: the audit cannot say which of the two images is the true duplicate, so it keeps neither.

On the current data this isolates 258 edges and 11 samples that were actually in an include manifest (11 train, 0 val). A further 394 endpoints named by those edges are already outside both manifests (`pending_audit` or `exclude`), so isolating them changes nothing in the manifests; they are counted here only so the two numbers are not conflated.

**This is NOT completed manual deduplication.** It is an isolation rule applied in the absence of manual deduplication. It guarantees that no known-suspected pair straddles the frozen split, and it pays for that guarantee by discarding samples that are probably fine. In the worst case a false-positive edge costs two real images; on this data most flagged endpoints were already outside the manifests, so the real cost was 11 images. Completing the human review will let most of these be returned.

