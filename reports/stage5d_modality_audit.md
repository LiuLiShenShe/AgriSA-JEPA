# Stage 5D - modality audit

The decisive evidence is the dataset structure, not the JPEG decode mode.
``agrifm_ref11`` publishes two sibling channel folders per phase:
``P{1,2}/rgb/<id>_rgbnormal_*.jpeg`` and ``P{1,2}/nir/<id>_nirraw_*.jpeg``.
The ``rgb``/``nir`` folder token is the dataset's own statement about the
spectrum of the file; a 3-channel JPEG proves nothing.

## Per-source sampled audit

| source | sampled | documented visible RGB | documented not visible | documented unknown | pixel screen pass | pixel screen fail |
|---|---:|---:|---:|---:|---:|---:|
| agrifield40k_pool | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref1 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref11 | 40 | 17 | 23 | 0 | 20 | 20 |
| agrifm_ref20 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref57 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref58 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref6 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref7 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref8 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref9 | 40 | 40 | 0 | 0 | 40 | 0 |
| agrifm_ref90 | 40 | 40 | 0 | 0 | 40 | 0 |
| applebbch81 | 40 | 40 | 0 | 0 | 40 | 0 |
| artemis_imagesafari | 40 | 40 | 0 | 0 | 40 | 0 |
| sesame_weed | 40 | 40 | 0 | 0 | 40 | 0 |

## agrifm_ref11 channel evidence (sampled)

- ``rgb`` folder images: **17** sampled, all decode as RGB,
  channel correlations ~0.94/0.98/0.96 (a photographic image).
- ``nir`` folder images: **23** sampled; mean channel values
  around (149.5, 1.7, 72.6) with corr(r,g) ~0.08 -- the green channel is
  essentially dead and the red channel carries NIR.  These are not
  visible-light RGB.

## Consequence

R3v2_strict_rgb admits only the ``rgb`` folder images of ``agrifm_ref11``.
Every ``nir`` image is excluded with primary reason
``MODALITY_NOT_VISIBLE_RGB``.
