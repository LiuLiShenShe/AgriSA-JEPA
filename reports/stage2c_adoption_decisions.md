# Stage 2C adoption decisions (per-prefix)

**These are RECOMMENDATIONS ONLY and require a named human researcher's final
approval.** Nothing here was executed: no row `status` changed, `data/frozen/`
was not regenerated, and no prefix was admitted. The upstream evidence already
recorded in [`stage2_source_index_evidence.md`](stage2_source_index_evidence.md)
still stands as-is.

The three buckets are:

- `recommended_for_adoption` — evidence and licence are strong enough that a
  curator could reasonably promote, with the conditions noted.
- `needs_additional_evidence` — the recorded evidence supports only a fraction
  of the prefix, or a reproduction gap exists.
- `recommended_for_exclusion` — adopt only with an explicit, separate decision.

## 1. How much of each prefix does the evidence actually cover?

Each prefix's `original_id` values are all distinct (one per image), so "how many
distinct IDs" is not the interesting question — the interesting question is how
many were **checked against the upstream index**. The recorded evidence names
only a handful of observed IDs per prefix:

| Prefix | Rows | Candidate source | Distinct original IDs | IDs checked vs upstream index | Evidence class | Upstream index total vs retained |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `gc` | 435 | GrassClover | 435 | **2** (`biomass_image_test_0100`, `..._0101`) | `original_source_index` | 435 = card count |
| `mw` | 255 | Maize-Weed | 255 | **2** (`C00020826_output`, `C00020908_output`) | `original_source_index` | 843 original, 255 retained |
| `pags` | 516 | PalmerAmaranth | 516 | **1** (`19345061_ext_cotton_20210825_125203`) | `original_source_index` | 614 original, 516 retained |
| `per` | 240 | PerennialPlants | 240 | **2** (`DSCF8400`, `DSCF8401`) | `original_source_index` | 392 original, 240 retained |
| `rl` | 809 | RumexLeaves | 809 | **2** (`10105`, `10122`) | `original_source_index` | 809 = card count |
| `rn` | 135 | Ronin | 135 | **2** (`12202`, `12229`) | `original_source_index` | 1,176 original, 135 retained |
| `saw` | 1,300 | SesameWeed | 1,300 | **2** (`agri_0_1009`, `agri_0_1017`) | `original_source_index` | 1,300 = card count |
| `vcd` | 2,258 | VCD | 2,258 | **1** (`2019-05-23_montoldre_bean_1_000000` + `.rf.<hex>` suffix) | `original_source_index` | 2,258 = card count |
| `weds` | 1,553 | WE3DS | 1,553 | **2** (`img_00100`, `img_00102`) | `original_source_index` | 2,568 original, 1,553 retained |
| `mc` | 7,242 | MuST-C | 7,242 | **components only** (nested path, no flattened literal) | `supporting_only` | 7,242 = card count |

**The honest generalisation statement.** For every prefix except `mc`, the
evidence establishes that the *observed original ID exists verbatim in the named
source's own published index* — for **1 or 2 IDs out of 135–2,258**. It does
**not** establish that every remaining ID in the prefix belongs to that source,
because no one enumerated the full upstream index against the full prefix. Where
the upstream total equals the retained count (`gc`, `rl`, `saw`, `vcd`, `mc`) the
count agreement is a supporting signal; where the retained count is a subset
(`mw`, `pags`, `per`, `rn`, `weds`) nothing local proves which subset was
filtered.

What would close the gap for a prefix: enumerate the full upstream index and
check every `original_id` in the prefix against it (e.g. a downloaded archive
central directory, a Kaggle/Mendeley/figshare full file listing), then record the
coverage percentage. Only `gc`, `rl`, `saw` and `vcd` have count-agreement that
makes the gap small.

## 2. Decision table

Licences below are the **publisher-card** values (<https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/7f530c67572b6a572467adab840cdbbd7029b112/README.md>);
see the honesty note in §5 about upstream re-verification.

| Prefix | Source | Licence | Evidence coverage | Overlap risk | Recommendation |
| --- | --- | --- | ---: | --- | --- |
| `gc` | GrassClover | **CC BY-SA 4.0 (copyleft)** | 2/435, count-agrees | Low (robotics-adjacent, used in segmentation benchmarks) | `needs_additional_evidence` — licence risk first |
| `mw` | Maize-Weed | CC BY 4.0 | 2/255 | Low | `recommended_for_adoption` (conditional) |
| `pags` | PalmerAmaranth | CC BY 4.0 | 1/516 | Low | `recommended_for_adoption` (conditional) |
| `per` | PerennialPlants | **MIT** | 2/240 | Low | `recommended_for_adoption` (conditional) |
| `rl` | RumexLeaves | CC BY 4.0 | 2/809, count-agrees | Low | `recommended_for_adoption` (conditional) |
| `rn` | Ronin | CC BY 4.0 | 2/135 | Low | `recommended_for_adoption` (conditional) |
| `saw` | SesameWeed | CC0 1.0 | 2/1,300, count-agrees | Low | `recommended_for_adoption` (conditional) |
| `vcd` | VCD | CC BY 4.0 | 1/2,258, count-agrees | Low | `recommended_for_adoption` (conditional) |
| `weds` | WE3DS | CC BY 4.0 | 2/1,553 | Low (used in weed-segmentation benchmarks) | `needs_additional_evidence` — reproduction gap (Zenodo 403) |
| `mc` | MuST-C | CC BY 4.0 | components only | Medium (7,242 rows = 47% of the pending corpus) | `needs_additional_evidence` — see §3 |

### Bucket counts
- `recommended_for_adoption` — **7** (`mw`, `pags`, `per`, `rl`, `rn`, `saw`, `vcd`)
- `needs_additional_evidence` — **3** (`gc`, `weds`, `mc`)
- `recommended_for_exclusion` — **0**

## 3. Conditions attached to the `recommended_for_adoption` bucket

Adoption is not a yes/no. Every prefix in that bucket should be promoted only
after all of the following, which is exactly the promotion path already
documented in `stage2_source_index_evidence.md`:

1. **Full-index enumeration.** Enumerate the upstream index and check *every*
   `original_id` in the prefix against it, recording a coverage percentage. The
   current evidence covers 1–2 IDs per prefix; that is a spot check, not proof.
2. **Licence re-verified upstream.** The publisher card is a secondary source for
   licences. Confirm at the source (Mendeley record page, Kaggle licence, figshare
   licence, GitHub repo).
3. **Fresh duplicate screen** over the newly admitted rows — they were never
   screened because they were never in an include manifest.
4. **New data version + re-freeze** (`agrifield40k-v1` or later), fresh SHA-256s.
5. **Recorded in the Stage 3 protocol** before any training run uses the split.

`vcd` additionally needs one decision: the observed `original_id` ends in
`.rf.<hex>` (a Roboflow export artefact), so the curator must record that the
`.rf.<hex>` suffix is stripped to recover the upstream stem.

## 4. Licence-specific flags

- **`gc` is CC BY-SA 4.0 (copyleft).** If GrassClover is confirmed, the derived
  images remain under CC BY-SA 4.0 and its attribution requirement propagates
  into the compilation's redistribution terms. The compilation is already
  CC BY-SA 4.0, so this does not change the compilation's own licence — but it
  means a redistribution must carry the Share-Alike obligation for the `gc`
  subset, and it is the reason `gc` is in `needs_additional_evidence` rather than
  `recommended_for_adoption`. Note PhenoBench is also CC BY-SA 4.0 but is
  excluded entirely, so `gc` would be the only Share-Alike source in the frozen
  split.
- **`per` is MIT.** Permissive; the least restrictive of the ten. No Share-Alike
  or attribution-on-redistribution complication beyond retaining the notice.
- **`saw` is CC0 1.0.** Public domain; no restriction at all.
- **`iNatWeeds is already in the frozen split** and carries `per_image_metadata`;
  its per-image attribution is covered in
  [`stage2c_data_gate.md`](stage2c_data_gate.md) §4.

## 5. Honesty note on evidence strength

- The per-prefix evidence is a **literal filename match for 1–2 IDs**, plus
  agreement on total counts for `gc`/`rl`/`saw`/`vcd`/`mc`. It is strong evidence
  that the prefix→source mapping is *plausible and correct*; it is not an
  enumeration of the whole prefix.
- Licences in §2 are the publisher card's values, re-read from the pinned
  revision. They were **not** re-verified at each upstream source in Stage 2C
  (network path restriction; see `stage2c_data_gate.md` §6).
- `weds` carries an explicit reproduction gap: Zenodo returned HTTP 403 during
  the earlier session, so its evidence rests partly on a derived re-host
  (HuggingFace `Voxel51/WE3DS`) plus the MDPI paper, not a direct re-read.
- `mc` is components-only and is analysed separately below.
- **No prefix was admitted.** Every row remains `status=pending_audit` and
  outside the frozen split.

## 6. `mc` / MuST-C mapping-rule analysis

`mc` is `supporting_only` and **stays `supporting_only`**. It is not upgraded to
confirmed here, and no `mc` row is admitted.

### 6.1 The two naming schemes

- **Observed AgriField ID (flattened):** `plot162_day230601_pos01_nikon_0`
- **Official BonnData MuST-C path (nested):** `images/UGV-RGB/230601/plot162/pos01/nikon_0.jpeg`
  (record <https://doi.org/10.60507/FK2/OX9XTM>, archive listing
  <https://bonndata.uni-bonn.de/api/access/datafile/8066>)

### 6.2 Component correspondence (one-to-one)

All 7,242 observed `mc` IDs match `^plot(\d+)_day(\d+)_pos(\d+)_nikon_(\d+)$`
exactly (0 exceptions):

| Observed component | Official path component | Correspondence |
| --- | --- | --- |
| `plot<NNN>` (34 distinct plots) | `plot<NNN>/` directory | one-to-one |
| `day<YYYYMMDD>` (5 dates: 230601, 230613, 230616, 230623, 230628) | `<YYYYMMDD>/` directory | one-to-one (date value) |
| `pos<NN>` (pos00–pos05) | `pos<NN>/` directory | one-to-one |
| `nikon_<N>` (0–19) | `nikon_<N>.jpeg` file stem | one-to-one |

### 6.3 The component with NO official counterpart

The literal token **`day`** in `day230601` has no official counterpart. The
official path stores the bare date `230601` as a directory name; AgriField
prepends the token `day` when flattening. So the mapping is
`<YYYYMMDD>` → `day<YYYYMMDD>`, an AgriField-side labelling convention, not an
upstream field. It is the one component the evidence cannot tie to the archive.

### 6.4 Is the reverse mapping (official path → observed ID) unique?

**Yes, on the observed set.** The flattening is field-delimited
(`plot`/`day`/`pos`/`nikon_`) and every field is numeric, so it is injective:
distinct official tuples cannot produce the same flattened string. Confirmed
directly — all **7,242** observed 4-tuples are distinct (0 collisions). So no two
different official paths flatten to the same observed ID.

One wrinkle worth recording: the official nested path orders fields
`date / plot / pos / nikon`, whereas the flattened ID orders them
`plot / day(date) / pos / nikon`. The flattening **reorders** `plot` ahead of the
date token; this is deterministic and reversible as long as the canonical layout
is known, but it means the reverse mapping is only unique *given that known
layout*, not from the string alone.

### 6.5 Is the observed ID set consistent with the archive's file count?

- Published retained count (card): **7,242**. Observed `mc` rows: **7,242**.
- The evidence stated the archive count is "consistent with the card retained
  count". Stage 2C could **not** independently recount the archive (no local copy).
- **Caveat that weakens the components-only claim:** the evidence enumerated the
  official archive components as `pos00`–`pos03` and dates `230601 / 230613 /
  230616`, but the observed data contains `pos00`–`pos05` and **two further dates
  (230623, 230628)**. So the evidence enumerated only a *subset* of the component
  space; it did not establish that every observed component exists in the archive.
- The possible tuple space is 34 plots × 5 dates × 6 positions × 20 cameras =
  **20,400**; the observed 7,242 is a sparse subset, which is consistent with the
  publisher's temporal de-duplication / quality filtering, but is not proof.

### 6.6 Conclusion and what would promote `mc`

**`mc` stays `supporting_only`.** Every component maps plausibly, but the
observed scheme is a flattened/reordered string with an AgriField-only `day`
token, not a literal filename match, and the evidence enumerated only part of the
component space. It is the largest single block of pending rows (7,242 = 47% of
the 14,743), so admitting it is the highest-impact decision of the ten.

**The artefact that WOULD promote it:** the publisher's own short-prefix → source
map (the AgriField-40K `mc` → MuST-C mapping and its flattening convention), or a
complete MuST-C archive listing enumerating every `<date>/plotNNN/posNN/nikon_N`
combination so that all 7,242 observed tuples can be checked one-to-one. Absent
that, `mc` remains a strongly-supported hypothesis, not a confirmed source.

## 7. Suggested order of work

1. Start with the `recommended_for_adoption` bucket, after enumerating the full
   upstream index for each prefix (§3 step 1).
2. Resolve `weds` reproduction (re-read Zenodo, or accept the derived-re-host
   evidence explicitly).
3. Decide `gc` licence handling (Share-Alike) and `mc` (largest, weakest evidence).
4. Only then bump `data_version` and re-freeze; never edit a frozen manifest in place.

