# Handoff: Penalized Likelihood Refactoring

## Date: 2026-01-24
## Phase: 5 (COMPLETED) - REFACTORING COMPLETE
## Status: All phases complete

---

## Project Context

We are refactoring the penalized likelihood fitting code in MultistateModels.jl to achieve correctness, transparency, robustness, and maintainability. The current implementation has several problems including fake solution objects, mixed concerns in _fit_exact, and tight coupling to quadratic penalties.

Branch: penalized_splines

---

## Required Reading (In Order)

1. scratch/PENALIZED_LIKELIHOOD_REFACTORING_PLAN.md - The master plan document (1,222 lines)
2. .github/skills/codebase-knowledge/SKILL.md - Codebase structure and conventions

---

## Key Architectural Decisions Already Made

1. HARD REQUIREMENT: All optimizations use Ipopt with ForwardDiff AD - no LBFGS, no finite differences
2. No backward compatibility needed - there are no external users
3. Final fit ALWAYS happens in _fit_exact_penalized, NOT in selection functions
4. Selection functions return HyperparameterSelectionResult with warmstart_beta, not fitted models
5. No fake solution tuples - all paths return real OptimizationSolution
6. Dispatch-based architecture using AbstractPenalty and AbstractHyperparameterSelector types

---

## Current State

Files modified in Phase 1:
- src/types/abstract.jl - Added AbstractPenalty, AbstractHyperparameterSelector
- src/types/penalties.jl - NEW FILE with penalty types, selector types, interface methods
- src/types/infrastructure.jl - Removed penalty types (now in penalties.jl)
- src/MultistateModels.jl - Added include for penalties.jl, updated exports

Files modified in Phase 2:
- src/inference/fit_common.jl - Added _resolve_selector() function
- src/inference/smoothing_selection.jl - Updated outdated comment (infrastructure.jl → penalties.jl)
- MultistateModelsTests/unit/test_penalty_infrastructure.jl - Added _resolve_selector tests

Files modified in Phase 3:
- src/inference/fit_penalized.jl - NEW FILE with _fit_exact_penalized, _fit_coefficients_at_fixed_hyperparameters
- src/inference/smoothing_selection.jl - Added _select_hyperparameters dispatcher and supporting functions
- src/MultistateModels.jl - Added include for fit_penalized.jl (before fit_exact.jl)
- MultistateModelsTests/unit/test_penalty_infrastructure.jl - Added Phase 3 integration tests

Files modified in Phase 4:
- src/inference/fit_exact.jl - Refactored to dispatcher pattern
- src/MultistateModels.jl - Removed select_smoothing_parameters from exports
- src/inference/smoothing_selection.jl - Added deprecation warnings to old functions

Files modified in Phase 5:
- src/inference/fit_exact.jl - Updated docstring to reflect dispatcher pattern
- src/inference/fit_penalized.jl - Added @warn for vcov_type=:model with penalized models
- src/inference/smoothing_selection.jl - Updated cross-references from deprecated to new functions

Tests passing: Yes (2097 passed, 0 errors)
Package loads: Yes

---

## Phase 1 Completed Actions

Action 1.1: ✅ Added abstract types to src/types/abstract.jl
- AbstractPenalty with docstring describing required interface
- AbstractHyperparameterSelector with docstring

Action 1.2: ✅ Created src/types/penalties.jl (NEW FILE)
- NoPenalty struct with interface methods
- QuadraticPenalty struct (same fields as old PenaltyConfig)
- PenaltyConfig const alias for QuadraticPenalty (backward compatibility)
- Selector structs: NoSelection, PIJCVSelector, ExactCVSelector, REMLSelector, PERFSelector
- HyperparameterSelectionResult struct
- Helper types: PenaltyTerm, TotalHazardPenaltyTerm, SmoothCovariatePenaltyTerm, SplinePenalty
- Full interface methods for all penalty types

Action 1.3: ✅ Updated src/types/infrastructure.jl
- Removed all penalty types (now in penalties.jl)
- Kept threading config and AD backend types

Action 1.4: ✅ Updated src/MultistateModels.jl
- Added include("types/penalties.jl") after infrastructure.jl
- Exported new types: AbstractPenalty, AbstractHyperparameterSelector, NoPenalty, QuadraticPenalty,
  NoSelection, PIJCVSelector, ExactCVSelector, REMLSelector, PERFSelector, HyperparameterSelectionResult,
  n_hyperparameters, get_hyperparameters, set_hyperparameters, hyperparameter_bounds

Action 1.5: ✅ Verified compilation and tests
- Package loads successfully
- All 29 Phase 1 type tests pass
- Full test suite: 2093 passed, 2 pre-existing errors (unrelated to this refactoring)

---

## Phase 2 Completed Actions

Action 2.1: ✅ Verified _resolve_penalty already exists
- _resolve_penalty() in fit_common.jl converts penalty kwarg to SplinePenalty or nothing
- This function was already implemented before Phase 2

Action 2.2: ✅ Created _resolve_selector helper function
- Added _resolve_selector(select_lambda::Symbol, penalty::AbstractPenalty) to fit_common.jl
- Maps symbols to AbstractHyperparameterSelector subtypes:
  - :none → NoSelection()
  - :pijcv, :pijlcv → PIJCVSelector(0)
  - :pijcv5, :pijcv10, :pijcv20 → PIJCVSelector(5/10/20)
  - :loocv → ExactCVSelector(0)
  - :cv5, :cv10, :cv20 → ExactCVSelector(5/10/20)
  - :efs → REMLSelector()
  - :perf → PERFSelector()
- Returns NoSelection() when penalty is NoPenalty
- Throws ArgumentError for unknown symbols

Action 2.3: ✅ Verified build_penalty_config returns QuadraticPenalty
- build_penalty_config() returns PenaltyConfig which is aliased to QuadraticPenalty
- No changes needed - the alias ensures compatibility

Action 2.4: ✅ Verified existing penalty code works with new types
- fit_exact.jl uses has_penalties() and PenaltyConfig correctly
- smoothing_selection.jl uses compute_penalty_from_lambda() with PenaltyConfig
- Updated outdated comment referencing infrastructure.jl → penalties.jl

Action 2.5: ✅ Added unit tests for _resolve_selector
- Added 26 tests to test_penalty_infrastructure.jl
- Tests cover all selector variants, NoPenalty edge case, and error handling

---

## Phase 3 Completed Actions

Action 3.1: ✅ Created src/inference/fit_penalized.jl (NEW FILE ~300 lines)
- _fit_exact_penalized() - Main penalized fitting entry point
  - Builds penalty config from penalty spec
  - Calls _select_hyperparameters to get HyperparameterSelectionResult
  - Calls _fit_coefficients_at_fixed_hyperparameters for final fit
  - Updates model parameters from solution
  - Computes vcov via _exactdata_vcov
  - Creates proper MultistateModelFitted object
- _fit_coefficients_at_fixed_hyperparameters() - Final optimization at fixed λ
  - Uses Ipopt via MathOptInterface (not LBFGS!)
  - Returns REAL OptimizationSolution object
  - Properly handles bounds and constraints

Action 3.2: ✅ Updated src/MultistateModels.jl
- Added include("inference/fit_penalized.jl") BEFORE fit_exact.jl
- Ensures fit_penalized functions are available when fit_exact loads

Action 3.3: ✅ Implemented _select_hyperparameters dispatcher
- Added to top of smoothing_selection.jl
- Dispatches on AbstractHyperparameterSelector subtypes:
  - NoSelection → immediate return with warmstart_beta
  - PIJCVSelector → _nested_optimization_pijcv
  - ExactCVSelector → _grid_search_exact_cv
  - REMLSelector → _nested_optimization_reml
  - PERFSelector → _nested_optimization_perf
- All selection functions return HyperparameterSelectionResult (NOT fitted models)

Action 3.4: ✅ Refactored nested optimization functions
- _nested_optimization_pijcv - Wood 2024 NCV method
- _grid_search_exact_cv - Grid search for exact CV (LOOCV, k-fold)
- _nested_optimization_reml - REML-based selection
- _nested_optimization_perf - PERF criterion selection
- _nested_optimization_criterion - Generic criterion optimizer
- All return HyperparameterSelectionResult with:
  - lambda: selected smoothing parameters
  - warmstart_beta: coefficient values for warm-starting final fit
  - penalty: updated penalty config
  - criterion_value, edf, converged, method, n_iterations, diagnostics

Action 3.5: ✅ Created _fit_inner_coefficients
- Inner loop coefficient fitting at fixed λ
- Uses Ipopt via MathOptInterface
- Returns fitted beta vector (not a solution object)
- Used by nested optimization for iterative updates

Action 3.6: ✅ Added Phase 3 integration tests
- _select_hyperparameters dispatcher tests (NoSelection)
- _fit_inner_coefficients tests (returns correct-sized finite vector)
- _fit_coefficients_at_fixed_hyperparameters tests (returns real OptimizationSolution)
- HyperparameterSelectionResult structure tests
- Fixed DataFrame column order issues in tests (id, tstart, tstop, statefrom, stateto, obstype)
- Fixed Hazard syntax (knots=[...] not nknots=...)

---

## Phase 3 Completion Criteria - ALL MET

- ✅ _fit_exact_penalized function implemented
- ✅ _fit_coefficients_at_fixed_hyperparameters uses Ipopt, returns real solution
- ✅ _select_hyperparameters dispatcher implemented for all selector types
- ✅ All selection functions return HyperparameterSelectionResult
- ✅ _fit_inner_coefficients implemented for nested optimization
- ✅ Integration tests pass
- ✅ Package loads without errors
- ✅ All tests pass (2093 passed, 2 pre-existing errors unrelated to refactoring)

---

## Phase 4 Completed Actions

Action 4.1: ✅ Updated _fit_exact to be a dispatcher
- Added dispatch logic at the top of _fit_exact
- Calls _fit_exact_penalized when penalty is active AND resolved_penalty != nothing
- Calls existing unpenalized MLE path otherwise

Action 4.2: ✅ Removed penalty-related code from old _fit_exact path
- Removed smoothing parameter selection code (was calling old select_smoothing_parameters)
- Removed fake solution object creation: `(u=, objective=, retcode=)` tuple
- Removed penalized likelihood function creation in the unpenalized path
- The old path now ONLY handles unpenalized MLE

Action 4.3: ✅ Verified fit() passes kwargs correctly
- fit() in fit_common.jl passes kwargs through to _fit_exact
- penalty, lambda_init, select_lambda all flow through correctly

Action 4.4: ✅ Legacy code cleanup
- Removed `select_smoothing_parameters` from exports in MultistateModels.jl
- Added deprecation warning to old select_smoothing_parameters function
- Added deprecation notice to second overload
- Updated docstring references to remove select_smoothing_parameters

Action 4.5: ✅ Tests verified
- Package loads successfully
- All 2097 tests pass (the 2 pre-existing errors are NOW FIXED)
- The old LBFGS path that was causing errors has been replaced with Ipopt

---

## Phase 4 Completion Criteria - ALL MET

- ✅ `_fit_exact` is now a clean dispatcher
- ✅ No fake solution objects anywhere (all paths return real OptimizationSolution)
- ✅ All existing tests pass (2097 passed, 0 errors)
- ✅ Penalized fitting goes through _fit_exact_penalized
- ✅ Unpenalized fitting uses clean MLE path
- ✅ Legacy select_smoothing_parameters marked deprecated

---

## Phase 5 Completed Actions (Cleanup and Polish)

Action 5.1: ✅ Analyzed deprecated function usage
- Verified: select_smoothing_parameters and fit_penalized_beta are only used by deprecated code paths
- Tests don't depend on them externally
- Decision: Keep deprecated functions for now (deprecation warnings in place), can be removed in future version
- Rationale: No need to rush removal since no external users and functions have deprecation warnings

Action 5.2: ✅ Updated docstrings
- Updated _fit_exact docstring to reflect dispatcher pattern (now documents dispatch logic, all kwargs, and references _fit_exact_penalized)
- Updated cross-references in smoothing_selection.jl:
  - compute_loocv_criterion: Changed reference from deprecated select_smoothing_parameters to _select_hyperparameters
  - compute_kfold_cv_criterion: Changed reference from deprecated select_smoothing_parameters to _select_hyperparameters
- Updated AD-SAFETY NOTES comment header in smoothing_selection.jl:
  - Changed fit_penalized_beta → _fit_inner_coefficients
  - Changed select_smoothing_parameters → _select_hyperparameters

Action 5.3: ✅ Added variance computation warning for penalized models
- Added @warn in _fit_exact_penalized when vcov_type=:model is used with penalties
- Warning explains that inverse Hessian of PENALIZED likelihood doesn't account for smoothing bias
- Recommends vcov_type=:ij (default) or vcov_type=:jk as alternatives

Action 5.4: ✅ Final test verification
- All 2097 tests pass
- Package loads successfully
- No errors or failures

Action 5.5: ✅ Updated documentation
- Updated this handoff document with Phase 5 completion
- Marked Phase 5 action items in PENALIZED_LIKELIHOOD_REFACTORING_PLAN.md

---

## Phase 5 Completion Criteria - ALL MET

- ✅ Deprecated functions analyzed and documented (kept with warnings)
- ✅ All functions have complete docstrings
- ✅ Variance computation warning added for penalized models
- ✅ All tests pass (2097 passed, 0 errors)
- ✅ Documentation updated

---

## Key Changes Summary (All Phases)

### New Files Created
- src/types/penalties.jl - Complete penalty type hierarchy and interface
- src/inference/fit_penalized.jl - Penalized fitting functions

### Files Modified
- src/types/abstract.jl - Added AbstractPenalty, AbstractHyperparameterSelector
- src/types/infrastructure.jl - Removed penalty types (moved to penalties.jl)
- src/inference/fit_common.jl - Added _resolve_selector()
- src/inference/fit_exact.jl - Refactored to dispatcher pattern, updated docstrings
- src/inference/smoothing_selection.jl - Added _select_hyperparameters dispatcher, updated references
- src/MultistateModels.jl - Updated includes and exports

### Architectural Changes
1. `_fit_exact` is now a clean dispatcher that routes to:
   - `_fit_exact_penalized` when penalties are active
   - Unpenalized MLE path otherwise
2. Selection functions return `HyperparameterSelectionResult`, NOT fitted models
3. Final fit ALWAYS happens in `_fit_coefficients_at_fixed_hyperparameters`
4. All optimizations use Ipopt with ForwardDiff (no LBFGS, no finite differences)
5. No fake solution tuples anywhere

### Tests
- All 2097 tests pass
- The 2 pre-existing spline calibration errors were FIXED by switching to Ipopt

---

## REFACTORING COMPLETE

All 5 phases have been completed successfully. The penalized likelihood fitting code has been refactored to achieve:

1. **Correctness**: No fake solution objects; real optimization throughout
2. **Transparency**: Clear separation between hyperparameter selection and coefficient fitting
3. **Robustness**: Consistent Ipopt+ForwardDiff optimization everywhere
4. **Maintainability**: Dispatch-based architecture enabling future penalty types

---

## Commands Reference

Load package: julia --project -e 'using MultistateModels'
Run tests: julia --project -e 'using Pkg; Pkg.test()'
Clear cache: rm -rf ~/.julia/compiled/v1.*/MultistateModels*
