# Resume Notes -- 2025-11-25

- Latest focused run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_modelgeneration"])'` (green, 178 tests).
- `censoring_panel_data()` now includes an obstype 3 row plus explicit `CensoringPatterns`; update any downstream consumers if they assumed unobstructed `stateto` values.
- Next FC2 increments:
  1. Port the remaining builder tests (hazards, helpers, surrogates) onto `TestFixtures` so coverage stays DRY.
  2. Add narrow unit tests for `build_totalhazards` and `build_emat` once fixture coverage lands.
  3. Wire the new fixtures into surrogate/likelihood tests before touching Tang simulation transforms.
- FC3 (ODE + simulation transforms) stays blocked until FC2 merges; open the `TBD-` GitHub issues listed in `docs/model_generation_testing.md` once you begin.
