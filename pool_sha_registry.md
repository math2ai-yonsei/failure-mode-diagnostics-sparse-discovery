# Pool SHA Registry

Each benchmark uses a fixed candidate trajectory pool. The 16-character SHA
fingerprints below are the authoritative, immutable identifiers for those
pools. All values are post-Savitzky-Golay-consistency patch: every pool
derivative and every training derivative is computed with the Savitzky-Golay
filter (no analytic derivatives).

A reviewer can confirm that a regenerated pool matches the paper by recomputing
its SHA fingerprint and comparing it against this table.

| Artifact                  | SHA (16 char)      | Used in                                       |
|---------------------------|--------------------|-----------------------------------------------|
| Cart-Pole ablation pool   | 16822e44cd5c00d1   | Cart-Pole D-optimal / random ablation         |
| AEK baseline pool         | 87594090343bee29   | AEK reparameterization analysis               |
| AEK dither pool           | fc5e11fa22f51172   | AEK coverage-recovered (dither-plus) pool     |
| AEK Reparam-2 theta_sha   | fe68a673ff1bd2e5   | AEK reparameterized library verification      |
| Lorenz-63 pool            | 8a884d874a2d7282   | Lorenz-63 augmentation analysis               |
| Lynx-Hare pool            | 44c855ee4b4a571c   | Lynx-Hare appendix analysis                   |
| Silverbox pool            | e5a590b95bb1b8dc   | Silverbox augmentation analysis               |

Notes:
- Pool SHAs are immutable once registered. They are never recomputed or
  overwritten; a mismatch indicates a divergent pool, not a registry update.
- The AEK Reparam-2 entry is a library (theta) fingerprint used to verify the
  reparameterized feature library, not a trajectory pool.
