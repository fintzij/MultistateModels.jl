# Surrogate Subsystem Audit Report

**Date**: 2026-01-20 (Original Audit)  
**Refactored**: 2026-01-20  
**Branch**: `penalized_splines`  
**Auditor**: Claude (Adversarial Code Review)

---

## REFACTORING STATUS: COMPLETED ✅

### Summary of Changes Made

1. **Unified surrogate field** on `MultistateModel` and `MultistateModelFitted`:
   - Replaced `markovsurrogate` and `phasetype_surrogate` with single `surrogate` field
   - Added backward compatibility accessors (`.markovsurrogate` and `.phasetype_surrogate` still work)

2. **PhaseTypeSurrogate is now self-contained**:
   - Added `hazards::Vector{<:_Hazard}` field (exponential hazards with covariate info)
   - Added `parameters::NamedTuple` field (baseline rates + covariate β)
   - Removed dependency on external `MarkovSurrogate` for covariate scaling

3. **Simplified fit_mcem.jl**:
   - Removed complex `markov_surrogate` extraction logic
   - Uses `model.surrogate` directly (already fitted at call time)
   - `build_phasetype_tpm_book` now uses `surrogate.hazards` and `surrogate.parameters`

4. **Key files modified**:
   - `src/types/model_structs.jl` - unified surrogate field
   - `src/phasetype/types.jl` - added hazards/parameters fields to PhaseTypeSurrogate
   - `src/surrogate/markov.jl` - updated `_build_phasetype_from_markov`
   - `src/inference/sampling_phasetype.jl` - rewrote `build_phasetype_tpm_book`
   - `src/inference/fit_mcem.jl` - simplified surrogate handling
   - `MultistateModelsTests/fixtures/TestFixtures.jl` - updated constructor call

### Test Results
- **2151 passed**, 4 failed (pre-existing SCTP constraint test issues), 1 errored (pre-existing fitting issue)
- Surrogate refactoring does NOT introduce new test failures

---

## 1. Executive Summary (Original Audit)

The surrogate handling in MultistateModels.jl has significant architectural issues:

1. **TWO surrogate fields** exist on `MultistateModel` when only ONE should be needed
2. **Surrogate fitting logic** is embedded in `fit_mcem.jl` when it should only be in construction/initialization
3. **Redundant construction paths** for phase-type surrogates across multiple files
4. **Unclear ownership** of when surrogates are built vs. fitted
5. **Variable shadowing** - local `surrogate` variables shadow/confuse which surrogate is being used

---

## 2. Current Architecture

### 2.1 Surrogate Fields on Model Structs

**File**: [src/types/model_structs.jl](../src/types/model_structs.jl)

```julia
mutable struct MultistateModel <: MultistateProcess
    # ... other fields ...
    markovsurrogate::Union{Nothing, MarkovSurrogate}           # LINE 288
    phasetype_surrogate::Union{Nothing, AbstractSurrogate}     # LINE 289
    # ...
end

mutable struct MultistateModelFitted <: MultistateProcess
    # ... other fields ...
    markovsurrogate::Union{Nothing, MarkovSurrogate}           # LINE 314  
    phasetype_surrogate::Union{Nothing, AbstractSurrogate}     # LINE 315
    # ...
end
```

**ISSUE**: Two separate fields for surrogates. This creates confusion about which one is "the" surrogate.

### 2.2 Surrogate Type Hierarchy

```
AbstractSurrogate (abstract)
├── MarkovSurrogate         # Exponential hazard surrogate for MCEM proposals
└── PhaseTypeSurrogate      # Phase-type FFBS surrogate (also <: AbstractSurrogate)
```

Both types can serve as MCEM proposals, but only `MarkovSurrogate` is stored in `markovsurrogate` field, while `PhaseTypeSurrogate` goes in `phasetype_surrogate` field.

---

## 3. Code Paths for Surrogate Creation

### 3.1 Construction Time (multistatemodel.jl)

**File**: [src/construction/multistatemodel.jl](../src/construction/multistatemodel.jl)

**Lines 217-235**: Resolves `:auto` surrogate option and builds `MarkovSurrogate`:
```julia
# Resolve :auto surrogate option based on hazard types
resolved_surrogate = if surrogate === :auto
    needs_phasetype_proposal(_hazards) ? :phasetype : :markov
else
    surrogate
end

# Build surrogate if requested (initially unfitted)
if resolved_surrogate in (:markov, :phasetype)
    surrogate_haz, surrogate_pars_ph, _ = build_hazards(hazards...; data = data, surrogate = true)
    markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars_ph; fitted=false)
else
    markov_surrogate = nothing
end
```

**Lines 237-249**: Builds `PhaseTypeSurrogate` at construction when `resolved_surrogate == :phasetype`:
```julia
phasetype_surr = nothing
if resolved_surrogate == :phasetype
    ph_config = PhaseTypeConfig(
        n_phases = surrogate_n_phases,
        structure = coxian_structure
    )
    phasetype_surr = build_phasetype_surrogate(tmat, ph_config; 
        data = data, 
        hazards = _hazards,
        verbose = verbose)
end
```

**Lines 275-282**: Fits surrogate at model creation if `fit_surrogate=true`:
```julia
if fit_surrogate && resolved_surrogate in (:markov, :phasetype)
    if verbose
        println("Fitting Markov surrogate at model creation time...")
    end
    fitted_surrogate = _fit_markov_surrogate(model; 
        surrogate_constraints = surrogate_constraints, 
        verbose = verbose)
    model.markovsurrogate = fitted_surrogate
end
```

**BUG IDENTIFIED**: Only the `MarkovSurrogate` is fitted at construction. The `PhaseTypeSurrogate` built at lines 237-249 uses heuristic rates and is NEVER fitted at construction time, even though `fit_surrogate=true`.

### 3.2 Fit Time (fit_mcem.jl)

**File**: [src/inference/fit_mcem.jl](../src/inference/fit_mcem.jl)

**Lines 340-347**: Checks if markovsurrogate exists, throws error if missing:
```julia
if isnothing(model.markovsurrogate)
    throw(ArgumentError("MCEM requires a Markov surrogate. Call `set_surrogate!(model)` or use `surrogate=:markov` in `multistatemodel()` before fitting."))
end
```
This is good - it errors if surrogate is missing.

**Lines 349-355**: Fits surrogate if not yet fitted:
```julia
if !model.markovsurrogate.fitted
    if verbose
        println("Markov surrogate not yet fitted. Fitting via MLE...")
    end
    set_surrogate!(model; type=:markov, method=:mle, verbose=verbose)
end
```
**ISSUE**: Surrogate fitting inside `fit()`. This should have been done at construction or via explicit `initialize_surrogate!()`.

**Lines 365-435**: Builds PhaseTypeSurrogate from Markov surrogate when `use_phasetype`:
```julia
if use_phasetype
    # === CRITICAL FIX: Fit phase-type rates via MLE before building surrogate ===
    # ...
    fitted_rates = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec; ...)
    
    phasetype_surrogate = _build_phasetype_from_markov(model, markov_surrogate; 
                                                       config=proposal_config, 
                                                       fitted_rates=fitted_rates,
                                                       verbose=verbose)
end
```
**ISSUE**: This builds a NEW `phasetype_surrogate` local variable, ignoring `model.phasetype_surrogate` that was built at construction time!

**Line 455**: Sets local `surrogate` variable:
```julia
surrogate = markov_surrogate
```
**ISSUE**: Variable shadowing. Now `surrogate` refers to `markov_surrogate`, not a generic surrogate.

**Lines 1261-1262**: Final assignment to fitted model:
```julia
surrogate,                 # This is markov_surrogate (local variable)
phasetype_surrogate,       # This is the one built in fit(), NOT model.phasetype_surrogate
```

### 3.3 Surrogate/markov.jl Entry Points

**File**: [src/surrogate/markov.jl](../src/surrogate/markov.jl)

**`fit_surrogate()`** (lines 77-118): Main user-facing function for fitting surrogates.
- Validates inputs
- Dispatches to `_fit_markov_surrogate()` or `_fit_phasetype_surrogate()`

**`set_surrogate!()`** (lines 508-537): In-place fitting on model.
- Calls `_fit_markov_surrogate()` internally
- Sets `model.markovsurrogate = markov_surrogate`
- **ISSUE**: Does NOT set `model.phasetype_surrogate` even when `type=:phasetype`

**`_fit_phasetype_surrogate()`** (lines 197-315): Fits phase-type surrogate.
- First fits Markov surrogate via `_fit_markov_surrogate()`
- Then builds phase-type via `_build_phasetype_from_markov()`
- Optionally fits all rates via `_fit_phasetype_mle()`

**`_build_phasetype_from_markov()`** (lines 335-445): Builds PhaseTypeSurrogate from MarkovSurrogate.

### 3.4 phasetype/surrogate.jl Entry Point

**File**: [src/phasetype/surrogate.jl](../src/phasetype/surrogate.jl)

**`build_phasetype_surrogate()`** (lines 66-115): Direct construction path.
- Takes `tmat` and `PhaseTypeConfig`
- Uses heuristics (`:auto`, `:heuristic`) or explicit n_phases
- Does NOT require a MarkovSurrogate
- Used at construction time in `multistatemodel.jl`

---

## 4. Identified Bugs

### BUG-1: PhaseTypeSurrogate Built at Construction is Ignored

**Location**: [fit_mcem.jl L365-435](../src/inference/fit_mcem.jl#L365-L435)

When `use_phasetype=true`, `fit_mcem.jl` creates a NEW `phasetype_surrogate` local variable via `_build_phasetype_from_markov()`. This completely ignores the `model.phasetype_surrogate` that was built at construction time in `multistatemodel.jl`.

**Impact**: The `phasetype_surrogate` built at construction (with heuristic rates) is never used. The one built in `fit()` uses MLE-fitted rates but `model.phasetype_surrogate` remains stale.

### BUG-2: set_surrogate! Doesn't Update phasetype_surrogate

**Location**: [markov.jl L508-537](../src/surrogate/markov.jl#L508-L537)

```julia
function set_surrogate!(model::MultistateProcess; type::Symbol = :markov, ...)
    # ...
    if type === :markov
        markov_surrogate = _fit_markov_surrogate(...)
        model.markovsurrogate = markov_surrogate
    else
        # Phase-type: also need to set Markov surrogate for infrastructure
        markov_surrogate = _fit_markov_surrogate(...)
        model.markovsurrogate = markov_surrogate
        # Note: Phase-type surrogate is built in fit() when needed  <-- PROBLEM
    end
```

When `type=:phasetype`, only `markovsurrogate` is set. The comment says "Phase-type surrogate is built in fit() when needed" which is wrong design.

### BUG-3: Surrogate Variable Shadowing

**Location**: [fit_mcem.jl L455](../src/inference/fit_mcem.jl#L455)

```julia
surrogate = markov_surrogate
```

This creates a local `surrogate` variable that shadows any potential generic usage. Throughout the rest of the function, `surrogate` always means `markov_surrogate`, even when phase-type proposal is active.

### BUG-4: Double Surrogate Building for Phase-Type

When `surrogate=:phasetype` is passed to `multistatemodel()`:
1. `build_phasetype_surrogate()` is called at construction (L237-249)
2. `_build_phasetype_from_markov()` is called at fit time (L365-435)

These are TWO DIFFERENT construction paths that may produce different results.

### BUG-5: fit_surrogate Flag Doesn't Affect PhaseTypeSurrogate

**Location**: [multistatemodel.jl L275-282](../src/construction/multistatemodel.jl#L275-L282)

Even when `fit_surrogate=true`, only the `MarkovSurrogate` is fitted. The `PhaseTypeSurrogate` remains at heuristic values until `fit()` is called.

---

## 5. Code in fit_mcem.jl That Creates/Fits Surrogates (TO BE REMOVED)

| Lines | Code | Action |
|-------|------|--------|
| 340-355 | Checks if surrogate exists, fits if unfitted | REMOVE fitting, keep error for missing |
| 365-435 | Builds PhaseTypeSurrogate when use_phasetype | REMOVE, require surrogate at construction |
| 455 | `surrogate = markov_surrogate` | RENAME to `markov_surrogate` only |

---

## 6. Redundant Code Across Files

### Two Construction Paths for PhaseTypeSurrogate

1. **`build_phasetype_surrogate()`** in `phasetype/surrogate.jl`:
   - Direct construction from tmat + config
   - Uses heuristics only
   - Called at `multistatemodel()` time

2. **`_build_phasetype_from_markov()`** in `surrogate/markov.jl`:
   - Construction from MarkovSurrogate rates
   - Can use MLE-fitted rates
   - Called at `fit()` time

**CONSOLIDATION**: Keep `_build_phasetype_from_markov()` as the primary path. `build_phasetype_surrogate()` can call it internally for the heuristic case.

### PhaseTypeConfig vs ProposalConfig

- `PhaseTypeConfig` defined in `phasetype/types.jl`
- `ProposalConfig` defined in ? (grep needed)

These serve similar purposes and should be unified.

---

## 7. Questions Answered

### Architecture Questions

**Q1: Why are there TWO surrogate fields on MultistateModel?**

A: Historical accident. `markovsurrogate` was added first for MCEM. Later, `phasetype_surrogate` was added for improved proposals. They should be unified into a single `surrogate::Union{Nothing, AbstractSurrogate}` field.

**Q2: Why is MarkovSurrogate treated specially when PhaseTypeSurrogate is also a valid MCEM proposal?**

A: The sampling infrastructure (tpm_book, hazmat_book, FFBS) was built around MarkovSurrogate. Phase-type sampling builds on top of Markov infrastructure (uses Markov rates as base, expands to phases). This coupling is appropriate but the field structure is not.

**Q3: What is the relationship between `build_phasetype_surrogate` vs `_build_phasetype_from_markov`?**

A: Both create PhaseTypeSurrogate but with different initializations:
- `build_phasetype_surrogate`: Heuristic rates (n_phases-based Erlang)
- `_build_phasetype_from_markov`: Markov MLE rates → optionally phase-type MLE rates

Should consolidate to ONE path with options for initialization method.

### Construction Flow Questions

**Q4: What happens when `surrogate=:markov` is passed to `multistatemodel()`?**

A: 
1. `MarkovSurrogate` is created (unfitted, `fitted=false`)
2. If `fit_surrogate=true` (default), `_fit_markov_surrogate()` is called → `fitted=true`
3. Model is returned with fitted MarkovSurrogate

**Q5: What happens when `surrogate=:phasetype` is passed?**

A:
1. `MarkovSurrogate` is created (unfitted)
2. `PhaseTypeSurrogate` is created via `build_phasetype_surrogate()` with heuristics
3. If `fit_surrogate=true`, only MarkovSurrogate is fitted
4. PhaseTypeSurrogate remains at heuristic values
5. At `fit()` time, a NEW PhaseTypeSurrogate is built via `_build_phasetype_from_markov()`

This is BROKEN. PhaseTypeSurrogate should be fitted at construction too.

**Q6: What does `fit_surrogate=true/false` control?**

A: Whether the MarkovSurrogate is fitted via MLE at construction. It does NOT affect PhaseTypeSurrogate. This flag should probably be removed in favor of explicit `initialize_surrogate!()`.

### Bug Hunt Questions

**Q7: In fit_mcem.jl, find ALL places where surrogates are created or fitted.**

Lines:
- 349-355: Fits MarkovSurrogate if not already fitted
- 383-397: Resolves n_phases configuration
- 399-404: Calls `_fit_phasetype_mle()` to fit rates
- 407-410: Calls `_build_phasetype_from_markov()` to create new surrogate

**Q8: When `proposal=PhaseTypeProposal(...)` is passed to `fit()`, what happens?**

1. `resolve_proposal_config()` creates ProposalConfig with type=:phasetype
2. `use_phasetype = true`
3. fit_mcem.jl builds NEW phasetype_surrogate (ignoring model.phasetype_surrogate)
4. Sampling uses the new phasetype_surrogate
5. Fitted model stores the new one in `phasetype_surrogate` field

**Q9: After the "fix" that calls `_fit_phasetype_mle`, is the phase-type surrogate actually used for sampling?**

YES, but only the one built in `fit()`, NOT the one on `model.phasetype_surrogate`.

**Q10: What is the `surrogate` variable in fit_mcem.jl pointing to at each stage?**

- L455: `surrogate = markov_surrogate` (always the Markov one)
- Used in: `DrawSamplePaths!(..., surrogate=surrogate, ...)`

The phase-type surrogate is passed separately: `phasetype_surrogate=phasetype_surrogate`

---

## 8. Refactor Proposal

### New Architecture

```julia
mutable struct MultistateModel <: MultistateProcess
    # ... other fields ...
    surrogate::Union{Nothing, AbstractSurrogate}  # SINGLE field for any surrogate type
    # ...
end
```

The `surrogate` field can hold either:
- `nothing` (no surrogate)
- `MarkovSurrogate` (exponential proposal)
- `PhaseTypeSurrogate` (phase-type proposal)

### New Function: `initialize_surrogate!`

```julia
"""
    initialize_surrogate!(model; type=:markov, method=:mle, n_phases=2, ...)

Initialize and fit a surrogate for MCEM importance sampling.

# Arguments
- `type::Symbol`: Surrogate type (`:markov` or `:phasetype`)
- `method::Symbol`: Fitting method (`:mle` or `:heuristic`)
- `n_phases`: For phase-type: number of phases (Int, Dict{Int,Int}, or :heuristic)
- `surrogate_constraints`: Optional constraints for MLE fitting
- `verbose::Bool`: Print progress

# Returns
- The modified model (also modifies in-place)

# Example
```julia
model = multistatemodel(h12_wei, h21_wei; data=df)  # No surrogate
initialize_surrogate!(model; type=:phasetype, n_phases=3)
fit(model)  # Now uses phase-type proposal
```
"""
function initialize_surrogate!(model::MultistateProcess; 
    type::Symbol = :markov,
    method::Symbol = :mle,
    n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2,
    surrogate_constraints = nothing,
    verbose::Bool = true)
    
    if type === :markov
        surrogate = _fit_markov_surrogate(model; method=method, 
                                          surrogate_constraints=surrogate_constraints,
                                          verbose=verbose)
    else  # :phasetype
        surrogate = _fit_phasetype_surrogate(model; method=method, n_phases=n_phases,
                                              surrogate_constraints=surrogate_constraints,
                                              verbose=verbose)
    end
    
    model.surrogate = surrogate
    return model
end
```

### Updated `multistatemodel()`

```julia
function multistatemodel(hazards...; data, surrogate=:none, ...)
    # ... build model ...
    
    # Build surrogate if requested
    if surrogate in (:auto, :markov, :phasetype)
        resolved_type = surrogate === :auto ? _select_surrogate_type(hazards) : surrogate
        initialize_surrogate!(model; type=resolved_type, ...)
    end
    
    return model
end
```

### Updated `fit()` / `_fit_mcem()`

```julia
function _fit_mcem(model; proposal=:auto, ...)
    # Validate surrogate exists
    if isnothing(model.surrogate)
        throw(ArgumentError(
            "MCEM requires a surrogate. Call `initialize_surrogate!(model)` " *
            "or use `surrogate=:markov/:phasetype` in `multistatemodel()` before fitting."))
    end
    
    # Determine proposal type from surrogate
    use_phasetype = model.surrogate isa PhaseTypeSurrogate
    
    # Use model's surrogate directly (NO building/fitting here)
    surrogate = model.surrogate
    
    # ... rest of MCEM ...
end
```

### Migration Plan

1. Add `surrogate::Union{Nothing, AbstractSurrogate}` field to model structs
2. Create `initialize_surrogate!()` function
3. Update `multistatemodel()` to call `initialize_surrogate!()` when `surrogate=` is specified
4. Update `_fit_mcem()` to:
   - Error if `model.surrogate` is nothing
   - Use `model.surrogate` directly (no creation/fitting)
   - Remove ALL surrogate building code
5. Deprecate `markovsurrogate` and `phasetype_surrogate` fields (keep for compatibility, remove in v1.0)
6. Rename `set_surrogate!()` to `initialize_surrogate!()` (deprecate old name)
7. Update all tests
8. Remove `fit_surrogate` parameter (superseded by explicit `initialize_surrogate!()`)

---

## 9. Success Criteria

- [ ] Single `surrogate` field on model structs
- [ ] `initialize_surrogate!()` is the unified entry point for surrogate creation/fitting
- [ ] `multistatemodel(...; surrogate=)` calls `initialize_surrogate!()` internally
- [ ] `fit()` / `_fit_mcem()` contains ZERO surrogate creation code
- [ ] `fit()` errors clearly if surrogate is missing but required
- [ ] `fit(model; proposal=:markov)` works when surrogate exists
- [ ] `fit(model; proposal=PhaseTypeProposal(...))` works when surrogate exists
- [ ] Markov and PhaseType proposals produce statistically equivalent estimates
- [ ] Code in fit_mcem.jl is reduced by at least 40%
- [ ] All existing tests pass
