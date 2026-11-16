# Verification — 2026-11-16

Validated on Windows with Python 3.12.14 and CPU PyTorch 2.15.0.

- `python -m pytest -q`: **52 passed**.
- `ruff check .`: no errors under the checked-in rule set.
- `ruff format --check .`: all Python sources formatted.
- `python tools/document_functions.py`: no functions missing docstrings,
  including private methods, constructors, nested functions and tests.
- `git diff --check`: no whitespace errors.
- `pip wheel --no-deps --wheel-dir artifacts .`: built `tri_mil-0.4.0`.
  The wheel includes model packages, configs, entrypoints and encoder metadata.

Tests cover all eight added models' classification forward/backward passes;
their CPU training entrypoints; NN_MIL masking, ensemble uncertainty and
prediction export; complete survival training/validation/unlabeled prediction;
Breslow Cox loss and gradients; censoring and C-index ties; PT/HDF5 features;
case splitting and leakage checks; retained ViT checkpoint helpers; YAML
override parsing; masked Nyström attention; and upstream TRIDENT state/summary
regressions, including Windows path statistics.

These are synthetic-feature and metadata regressions. GPU execution, real WSI
processing, pretrained encoder downloads and clinical predictive performance
were not validated in this environment. The GitHub Actions workflow also
defines Python 3.10/3.12 CPU checks; those hosted results are separate from the
local verification above.
