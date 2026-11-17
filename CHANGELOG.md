# Changelog

## 0.4.0 — 2026-11-16

- Integrate the local MIL_BASELINE snapshot's AS_MIL, ATTRI_MIL, CAR_MIL,
  FC_MIL, LIN_MIL, MR_AB_MIL, NN_MIL and PNEA_MIL, with YAMLs, model factory
  entries, dedicated classification training and case-level split scripts.
- Apply upstream fixes to TransMIL, ADD-MIL, AEM-MIL, AMD-MIL, CLAM-MB and SC-MIL.
- Sync TRIDENT 0.3.2 source: configurable patch-encoder image size,
  GigaPath Flash, Virchow2 CLS, Phaet and Mascaret encoders, BioIO image
  conversion, and reader, processing, state and summary fixes. Retain Tri-MIL's
  ViT-S encoder and inference dataset contracts.
- Add full-cohort Cox MIL survival training, validation C-index checkpoint
  selection, censoring-aware evaluation, unlabeled risk prediction, grouped
  survival CSV preparation and documented example configurations.
- Add function docstrings throughout the Python source tree, consistent Ruff
  formatting, documentation coverage checks and regression tests.
- Update English and Chinese README files, installation metadata and dependency
  constraints; record upstream input hashes in `docs/upstream-sync.json`.

Upstreams were supplied as local source snapshots without Git metadata; their
commit SHAs cannot be inferred. Model-weight downloads and full WSI integration
require external weights, native readers and representative slide data.
