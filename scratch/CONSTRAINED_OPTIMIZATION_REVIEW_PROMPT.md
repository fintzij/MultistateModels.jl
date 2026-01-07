# Review Prompt: Constrained Optimization Implementation Plan

## Role

You are a skeptical expert mathematician and senior Julia developer. Your task is to conduct a thorough adversarial review of the implementation plan and handoff document for the constrained optimization refactor.

## Context

We are removing the dual estimation/natural scale architecture from MultistateModels.jl. The goal is to:
1. Store all parameters on **natural scale only** (no log-transforms)
2. Use Ipopt box constraints (lb ≥ 0) for non-negativity instead of exp() transforms
3. Make the penalty function truly quadratic: P(β) = (λ/2)βᵀSβ
4. This fixes PIJCV λ selection which requires a quadratic objective

## Documents to Review

1. `scratch/CONSTRAINED_OPTIMIZATION_IMPLEMENTATION_PLAN.md`
2. `scratch/CONSTRAINED_OPTIMIZATION_HANDOFF_20260106.md`

## Your Review Must

### 1. Terminology Audit

Find ALL instances where the plan still uses:
- `exp()` or `exp.()` in code that should be removed (not in "TO BE REMOVED" context)
- "simplify" instead of "remove" when referring to exp-transform elimination
- "estimation scale" in new code (should only appear in "current/old" sections)
- `use_constraints` flags (decision was: no backward compatibility, remove entirely)
- `exp_transform` in new code (should only appear in "remove" context)
- Any conditional logic like `constrained ? x : exp.(x)` in proposed code

### 2. Mathematical Consistency

Verify that:
- The penalty is truly quadratic after changes: P(β) = (λ/2)βᵀSβ with no exp()
- Monotone spline formula uses `θ[i]` not `exp(θ[i])` for increments
- All hazard parameter storage is on natural scale
- PIJCV projection uses `max.(beta_loo, 0.0)` (non-negativity, not positivity)

### 3. Code Template Correctness

Check all code blocks in the plan to ensure:
- No exp() transforms remain in "AFTER" or "NEW" code
- No `constrained::Bool` parameters in function signatures
- Bounds are `lb = 0` not `lb = 1e-10` or `POSITIVE_LB`
- The `_spline_ests2coefs` function returns `ests` directly, not conditionally

### 4. Completeness Check

Verify the plan addresses removal of:
- `transform_baseline_to_estimation` function
- `transform_baseline_to_natural` function  
- `exp_transform` field from `PenaltyTerm` struct
- All exp() calls in `_spline_ests2coefs`
- All exp() calls in `compute_penalty`
- The dual `flat`/`natural` parameter storage

### 5. Consistency Between Documents

Check that the implementation plan and handoff document agree on:
- Lower bound value (should be 0, not 1e-10)
- No backward compatibility mode
- Dict-only API for user bounds (no Vector option)
- All terminology (non-negativity vs positivity)

### 6. Code Quality & Simplification Opportunities

This refactor is an opportunity to improve maintainability and robustness. Identify:

**Antipatterns to Resolve**:
- Redundant data storage (e.g., `flat` and `natural` storing the same thing)
- Conditional logic that can be removed entirely (e.g., `exp_transform ? ... : ...`)
- Functions that become identity operations and should be removed, not "simplified"
- Dead code paths that exist only for backward compatibility we're not keeping

**Simplification Opportunities**:
- Functions that can be removed entirely vs. functions that need modification
- Struct fields that become unnecessary
- Module exports that are no longer needed
- Test infrastructure that tests removed functionality

**Redundancies to Eliminate**:
- Duplicate parameter storage (estimation scale vs natural scale)
- Transform functions that become no-ops
- Flags/options that no longer have multiple valid values
- Documentation for removed features

**Maintainability Improvements**:
- Places where the plan says "simplify" but should say "remove"
- Code that would be clearer if restructured during this change
- Opportunities to reduce the public API surface
- Places where removing complexity now prevents future bugs

For each opportunity, note whether the plan already addresses it or if it's missing.

## Output Format

For each issue found, provide:
```
ISSUE #N: [Brief title]
Location: [File and line number or section]
Problem: [What's wrong]
Should be: [The correct version]
```

At the end, provide a summary count of issues by category.
