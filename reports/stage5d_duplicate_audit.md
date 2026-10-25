# Stage 5D - duplicate audit

- Ledger rows: 243,694
- Exact duplicate images removed: 9,384
- Near-duplicate candidate pairs: 82,474
- Confirmed near-duplicate pairs: 15,806
- Uncertain pairs kept in quarantine: 66,668
- Cross-source confirmed pairs: 2,188
- Cross-split confirmed pairs: 375 (both endpoints accepted: 0)
- Confirmed pairs touching a holdout image: 0
- Largest confirmed cluster: 36 images
- Cluster size distribution: {'2': 9201, '3-5': 404, '6-10': 161, '11-50': 51, '>50': 0}

Exact representatives are chosen by quality first (laplacian variance, pixel area, image_id), so the retained copy is a
pure function of the ledger, not of file order.

A hash neighbourhood is only a *candidate*.  Confirmation requires both
pHash and dHash agreement plus a DINOv2 cosine of at least 0.98;
everything else is quarantined rather than silently deleted.  Identical
pHash values over a texture set are caught as a giant cluster.

