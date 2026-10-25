# Stage 5D final report

## Status: **READY_AUTO_VALIDATED**

- TARGET_EXCEEDED: True (a scale property; it never upgrades a failed gate)
- All gates passed: True
- Failed gates: none
- Unmeasured gates: none

## Counts

- Ledger rows: 243,694
- Accepted: 169,326 (train 147,279 / val 22,047)
- Quarantined: 42,085
- Rejected: 9,301
- Holdout: 9,606
- Licence blocked: 2,500
- Duplicate removed: 10,876
- Scale target: 147,285; exceeded: True; shortfall: 0

## Gate detail

| gate | passed | blocking |
|---|---|---|
| files_readable | True | True |
| policy_and_iqa_scale_verified | True | True |
| visible_rgb_evidenced | True | True |
| field_domain_evidenced | True | True |
| licence_clear | True | True |
| dedup_resolved | True | False |
| holdout_screened | True | True |
| groups_disjoint_and_unknown_flagged | True | False |
| counts_conserved | True | True |
| rebuild_byte_identical | True | False |
| composition_from_manifest | True | False |
| legacy_unchanged | True | True |

## Hashes

- policy.yaml: `f89e1a4515c4f7353618c73da77bebf77d1b78e570266a4349177f7fd2bfbc97`
- train CSV: `f33b7cc24eee2d04e4f8b53d6b1026cfebe8fac5854971377216f3edb37abc30`
- val CSV: `a64a1222b27ddbdaa835a5c836a5330ceb1a71d1e18f0392f3f0b10a4d2e22d2`

## Status meaning

`READY_AUTO_VALIDATED` = all twelve programmatic gates passed.  It does
not mean human-verified, and it does not mean PROVEN_NO_LEAKAGE for
sources that are not on this machine.
