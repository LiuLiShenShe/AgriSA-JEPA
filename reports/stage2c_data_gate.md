# Stage 2C data gate

**Status: the `agrifield40k-v0` freeze STAYS `PROVISIONAL`.** No row was
auto-admitted. This document records what was checked in Stage 2C, the four
distinct risk buckets, the iNatWeeds / sequence-split findings, and what remains
for a human curator. It is a gate report, not an approval.

Scope note: Stage 2C is read-only with respect to every Stage 1 / Stage 2
artefact. The frozen manifests (`data/frozen/`), `data/duplicate_review.csv`, and
every row `status` are unchanged by this work. Adoption decisions here are
**recommendations** that require a named human researcher's sign-off.

## 1. Gate state

| Gate condition | State |
| --- | --- |
| 14,743 `pending_audit` rows resolved | no (evidence recorded, not adopted) |
| 784 duplicate candidates adjudicated | no (0 reviewed) |
| iNatWeeds per-image attribution/observation redistributable | retained, not re-verified |
| publisher sequence-level split independently verified | no |
| **Overall** | **PROVISIONAL — `gate_met: false`** |

The freeze remains the one produced by Stage 2 (`data_version=agrifield40k-v0`,
train 12,447 / val 3,145, train ∩ val = ∅). See [`data_freeze.md`](data_freeze.md)
for SHA-256 and the full split record.

## 2. Four explicitly distinct risk buckets

### (i) Candidates already conservatively isolated
- 258 edges (247 cross-split + 11 manifest-affecting) triggered the Stage 2B
  isolation rule; both endpoints were dropped from the frozen split where present.
- Of those endpoints, **11** were actually inside an include manifest (11 train,
  0 val); the other **394** were already outside both manifests
  (`pending_audit` / `exclude`), so isolating them is a no-op for the manifests.
- The isolation is a *policy*, not finished deduplication: it guarantees no
  *flagged* pair straddles the frozen split, and pays for that by discarding
  samples that are probably fine.

### (ii) Candidates not yet adjudicated
- All **784** edges are still `pending_manual_review`; **0** carry a human verdict.
- 776 perceptual candidates + 8 exact (sha256) edges. No reviewer name, no
  verdict, no `reviewed_at` is pre-filled anywhere.
- Tooling to work through them is `tools/duplicate_review_workflow.py`
  (`status` / `record` / `check` / `impact` / `refreeze-plan`). The tool refuses
  to write a verdict itself and refuses a blank or AI-sentinel reviewer.

### (iii) Risks that actually threaten an exploratory pilot on the current confirmed subset (15,592 images)

The honest answer, stated two ways:

- **No *known* leakage sits inside the frozen train/val as it stands.** Every one
  of the 247 cross-split candidate edges has **both** endpoints outside the
  include manifests (488 endpoints are `pending_audit`, 6 are PhenoBench), and
  every manifest-affecting edge's manifest endpoint was isolated out of the
  frozen split. So none of the 784 screener-flagged pairs currently straddles
  frozen train and frozen val.
- **But "no known" is not "none".** The screener (pHash ≤ 4 AND dHash ≤ 6) is a
  bound, not a proof, and the two checks below show the frozen split does *not*
  cleanly separate acquisition time. For an **exploratory pilot** this is
  acceptable if the pilot's numbers are labelled exploratory; for anything that
  gets reported as a clean generalisation result it is not.

Two concrete, locally-verifiable leaks found in Stage 2C (see §4):
1. **ACRECropWeed is a single robot capture sequence split randomly, not by
   sequence.** The nearest frozen train and frozen val timestamps are **2
   seconds apart**; 112 of 120 val frames have a train frame within 60 s.
2. **31 iNatWeeds *observations* (i.e. plants/occurrences) have images in both
   frozen train and frozen val.** Pixel-identity screening cannot see this
   because the images differ; they share provenance, not pixels.

Neither (1) nor (2) is caught by the duplicate screener. They are the strongest
argument against treating the current freeze as leak-free.

### (iv) Risks that MUST be re-checked if new sources are admitted later
- The 14,743 `pending_audit` rows were **never in an include manifest**, so
  duplicate screening was **never run on them**. Admitting any of the 10 prefixes
  requires a **fresh duplicate screen** over the newly admitted rows before they
  enter any split.
- The 10 prefixes would add ~14,743 images (frozen corpus 15,592 → ~30,000),
  which is a **data-version bump**, not a flag flip. It also changes the Stage 3 /
  Stage 4 comparison basis.
- Admitting `gc` (GrassClover, CC BY-SA 4.0) introduces a **copyleft** source
  into the compilation — see the adoption table for the redistribution
  implication.
- The publisher's 80/20 split for the pending prefixes is likewise unverified per
  image; admitting them imports the same sequence-split uncertainty (§4).

## 3. Verdicts and non-verdicts

No human review was performed in Stage 2C. Nothing in this document is a review
verdict, and nothing here changes a row's `status` or the frozen split. The
"recommended" buckets in [`stage2c_adoption_decisions.md`](stage2c_adoption_decisions.md)
are **suggestions requiring a named human researcher's final approval**.

## 4. iNatWeeds per-image attribution

iNatWeeds is licensed `per_image_metadata` in the Stage 1 registry (the
publisher card states CC BY 4.0 and says `iNatWeeds_metadata.json` carries the
required attribution). What is actually present in `data/all_samples.csv`:

| Field | Non-empty rows (of 1,315) | Coverage |
| --- | ---: | ---: |
| `attribution` | 1,315 | 100.0% |
| `observation_url` | 1,315 | 100.0% |
| `license` | 1,315 | 100.0% |
| ... of which in the frozen split | 1,309 | 100.0% of frozen |

- Attribution strings are real per-image credits, e.g.
  `(c) Sam Kieschnick, some rights reserved (CC BY)`; the CC0 subset reads
  `no rights reserved`.
- Licence split: **CC-BY-4.0 = 979**, **CC0-1.0 = 336**.
- `observation_url` is a live iNaturalist observation URL, e.g.
  <https://www.inaturalist.org/observations/308673477>.
- `data/iNatWeeds_metadata.json` is a 1,315-record list keyed by `photo_id` /
  `file`, each with `observer`, `license`, `attribution`, `observation_id`,
  `observation_url`, `taxon`. The `all_samples` join key is the numeric suffix of
  `original_id` (`<species>_<photo_id>`).

**What IS present:** every frozen iNatWeeds row already carries a licence, an
attribution string and an observation URL. So the earlier framing that these
fields are "unresolved" overstates the gap — the *fields are populated for
100% of rows*. What is missing is independent re-verification: Stage 2C did
**not** re-query the live iNaturalist API to confirm each attribution string is
still accurate, and the recorded `source_url` points at the compilation card,
not the individual observation (the per-image `observation_url` covers that).

**What would settle it:** a scripted join of `iNatWeeds_metadata.json` against
the iNaturalist observations API (or a frozen dump of it), asserting each
`observation_id` still resolves and its observed licence matches the recorded
one. Not done here (network egress from this host is restricted; see §6).

## 5. Sequence-level split — what can and cannot be verified locally

The publisher states the split is at sequence level for sequence-derived sources.
Stage 2C tested what the local artefacts actually support:

| Source | Local sequence proxy | Finding |
| --- | --- | --- |
| ACRECropWeed | `original_id` = `rgb-YYYY-MM-DD-HH-MM-SS`, all one date (2022-10-06) | **Refutes a strict sequence split.** Nearest frozen train↔val timestamps are **2 s apart**; 112/120 val frames have a train frame within 60 s. The robot sequence was split near-randomly. |
| iNatWeeds | `observation_id` (one observation = one plant), `photo_id` | **31 observations straddle frozen train and frozen val** (1,227 distinct observations, 1,309 photos). Photo IDs never straddle (0), so this is provenance-level, not pixel-level. |
| LUCASVision | 12 `lvb*` prefixes; `original_id` = `LUCAS<year>_<id>_Cover` | **Cannot be verified.** The `lvbNN` prefix does **not** encode acquisition year (e.g. `lvb11` contains all of 2006/2009/2012/2015/2018); there is no local time ordering to reconstruct a sequence. LUCASVision is listed as an independent-image source on the card, so the sequence rule arguably does not apply — but this is the publisher's claim, not something local data proves. |
| VCD / MuST-C (pending) | nested date path (Mendeley archive) | Not in the frozen split; unscreened. |
| Remaining prefixes | no shared sequence token in `original_id` | Not verifiable from local artefacts. |

**What CANNOT be verified from local artefacts:** the publisher's internal
sequence grouping for sources whose IDs carry no time/sequence token. Local data
has no sequence identifier for those, so no local check can confirm or refute the
80/20-at-sequence-level claim.

**External artefact that would settle it:** the publisher's own split-generation
manifest (a per-image `sequence_id → split` map), or a reproduction of their
pipeline. Absent that, the two findings above (ACRECropWeed seconds-apart;
31 iNatWeeds observations straddling) are the strongest local evidence that the
sequence rule was **not** uniformly honoured.

## 6. Web-evidence availability (honesty note)

The mandated `open-websearch` path could not be used from this host: the local
DNS resolver maps public hosts to `198.18.0.83`, which the skill classifies as a
private/loopback target and refuses ("Request URL resolves to a private or local
network target"), and its `search` subcommand returned only unrelated results.
Direct `curl` egress works, so the pinned publisher card was fetched and read
verbatim:

- <https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/7f530c67572b6a572467adab840cdbbd7029b112/README.md>
  — confirms compilation licence CC BY-SA 4.0; GrassClover CC BY-SA 4.0;
  PerennialPlants MIT; Sesame&Weed CC0 1.0; and that `iNatWeeds_metadata.json`
  provides the required attribution.
- A live search of the upstream GrassClover / MuST-C pages beyond the pinned
  card **was not completed**; those licences in
  [`stage2c_adoption_decisions.md`](stage2c_adoption_decisions.md) are the
  publisher-card values, not independently re-verified upstream.

## 7. What remains for a human researcher

1. Adjudicate the 784 duplicate candidates (minimum bar: the 258 cross-split /
   manifest-affecting edges) using `tools/duplicate_review_workflow.py record`.
2. Decide the per-prefix adoption buckets in
   [`stage2c_adoption_decisions.md`](stage2c_adoption_decisions.md) and record an
   explicit prefix→source map.
3. Decide how to handle the two concrete leaks in §2(iii) — re-split
   ACRECropWeed by timestamp and iNatWeeds by observation — before any reported
   result.
4. Re-verify iNatWeeds attributions against the observation API (§4), and
   re-check upstream licences directly rather than from the card (§6).
5. Only then bump `data_version`, regenerate, and re-record SHA-256.

**Nothing in Stage 2C is a promotion.** The freeze stays `PROVISIONAL`, the gate
stays unmet, and no row was auto-admitted.

