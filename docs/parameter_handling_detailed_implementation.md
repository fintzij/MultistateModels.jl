# Parameter Handling: Detailed Implementation Plan
**Date**: December 7, 2024  
**Status**: DETAILED SPECIFICATION - Ready for review

---

## Overview

This document provides line-by-line implementation details for converting MultistateModels.jl from positional parameter vectors to named parameter handling using ParameterHandling.jl's NamedTuple structure.

**Key Principle**: Leverage existing ParameterHandling.jl infrastructure - use nested NamedTuples for baseline parameters instead of anonymous vectors.

**Timeline**: 3 phases over 4-5 weeks, 35-50 hours total estimated effort.

---

## Phase 1: Foundation - Store Parameter Names and Convert to NamedTuples

**Goal**: Add parameter names to runtime objects and convert baseline/covariate parameters from vectors to NamedTuples.

**Estimated Effort**: 12-15 hours

**Deliverables**:
- Parameter names stored in hazard objects at runtime
- `build_hazard_params` returns nested NamedTuples
- ParameterHandling.jl compatibility verified
- Helper functions updated for extraction
- All existing tests pass unchanged

**Risk**: Low - purely structural changes, no behavioral modifications

---

### Task 1.1: Add `parnames` Field to Hazard Structs

**Duration**: 3-4 hours

**Objective**: Store parameter names in hazard objects so they're available at runtime.

#### Changes Required

**File: `src/common.jl`**

**Location 1: Lines 224-237 - `SemiMarkovHazard` struct**

Current:
```julia
struct SemiMarkovHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    emat::Union{Nothing, AbstractMatrix{Float64}}
end
```

Add after line 224:
```julia
struct SemiMarkovHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    parnames::Vector{Symbol}        # NEW: [:shape, :scale, :age, ...] (short names)
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    emat::Union{Nothing, AbstractMatrix{Float64}}
end
```

**Location 2: Lines 239-252 - `MarkovHazard` struct**

Current:
```julia
struct MarkovHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
end
```

Add after line 239:
```julia
struct MarkovHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    parnames::Vector{Symbol}        # NEW
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
end
```

**Location 3: Lines 265-281 - `PhaseTypeHazard` struct**

Current:
```julia
struct PhaseTypeHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    nphases::Int64
    progression_param_indices::Vector{Int64}
    exit_param_indices::Vector{Int64}
    # ... more fields ...
end
```

Add after line 265:
```julia
struct PhaseTypeHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    parnames::Vector{Symbol}        # NEW
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    nphases::Int64
    progression_param_indices::Vector{Int64}
    exit_param_indices::Vector{Int64}
    # ... rest unchanged ...
end
```

**Location 4: Lines 284-297 - `SplineHazard` struct**

Current:
```julia
struct SplineHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    # ... more fields ...
end
```

Add after line 284:
```julia
struct SplineHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    parnames::Vector{Symbol}        # NEW
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    # ... rest unchanged ...
end
```

#### Update Hazard Constructors

**File: `src/modelgeneration.jl`**

Each hazard builder function needs to:
1. Generate short parameter names (without hazard prefix)
2. Pass `parnames` to hazard constructor

**Location 1: Lines 276-335 - `_build_weibull_hazard` / `_build_gompertz_hazard`**

Current (lines 314-321):
```julia
# Generate parameter names
if hazfamily == :weibull || hazfamily == :gompertz
    parnames = [Symbol(string(ctx.hazname), "_shape"), 
                Symbol(string(ctx.hazname), "_scale")]
else
    parnames = [Symbol(string(ctx.hazname), "_intercept")]
end
append!(parnames, covariate_names)
```

Change to generate SHORT names:
```julia
# Generate SHORT parameter names (no hazard prefix)
if hazfamily == :weibull || hazfamily == :gompertz
    baseline_parnames = [:shape, :scale]
else
    baseline_parnames = [:intercept]
end

# For covariates, keep the original names (e.g., :age, :sex)
# Extract just the variable part from prefixed names like :h12_age → :age
covar_short_names = [Symbol(replace(string(name), r"^h\d+_" => "")) 
                     for name in covariate_names]

parnames = vcat(baseline_parnames, covar_short_names)
```

**Location 2: Lines 360-365 - Hazard constructor call**

Current:
```julia
hazard = SemiMarkovHazard(
    collect(1:length(ctx.hazmat_indices)),
    ctx.nstates,
    hazfuns,
    cumhazfuns,
    dhazfuns,
    dcumhazfuns,
    npar_baseline,
    npar_tv,
    npar_total,
    ctx.emat
)
```

Add `parnames` parameter:
```julia
hazard = SemiMarkovHazard(
    collect(1:length(ctx.hazmat_indices)),
    ctx.nstates,
    hazfuns,
    cumhazfuns,
    dhazfuns,
    dcumhazfuns,
    parnames,           # NEW: pass parameter names
    npar_baseline,
    npar_tv,
    npar_total,
    ctx.emat
)
```

**Repeat for all hazard types**: MarkovHazard (line ~380), PhaseTypeHazard (line ~450), SplineHazard (line ~500)

#### Testing

**File: `test/test_helpers.jl` (add new test)**

```julia
@testset "Hazard parnames storage" begin
    # Build simple 2-state Weibull model with covariate
    using StatsModels
    
    tmat = [0 1; 0 0]
    dat = DataFrame(id=1:10, tstart=0.0, tstop=1.0, age=rand(10))
    
    model = multistatemodel(
        @formula(time ~ age),
        dat,
        tmat,
        hazards = [Weibull(:PH)]
    )
    
    # Check parnames stored in hazard
    hazard = model.hazards[1]
    @test hasfield(typeof(hazard), :parnames)
    @test hazard.parnames == [:shape, :scale, :age]
    
    # Verify short names (no h12_ prefix)
    @test !any(startswith.(string.(hazard.parnames), "h"))
end
```

---

### Task 1.2: Modify `build_hazard_params` to Return NamedTuples

**Duration**: 4-5 hours

**Objective**: Convert baseline and covariate parameters from vectors to NamedTuples.

#### Changes Required

**File: `src/helpers.jl`**

**Location: Lines 266-283 - `build_hazard_params` function**

Current implementation:
```julia
function build_hazard_params(
    log_scale_params::AbstractVector{T}, 
    npar_baseline::Int,
    npar_total::Int
) where {T<:Real}
    
    # extract baseline parameters
    baseline = log_scale_params[1:npar_baseline]
    
    if npar_total > npar_baseline
        covariates = log_scale_params[(npar_baseline+1):npar_total]
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

New implementation:
```julia
"""
    build_hazard_params(log_scale_params, parnames, npar_baseline, npar_total)

Build nested NamedTuple structure for hazard parameters.

# Arguments
- `log_scale_params`: Vector of parameter values (log scale for baseline)
- `parnames`: Vector of parameter names (short names: :shape, :scale, :age, etc.)
- `npar_baseline`: Number of baseline parameters
- `npar_total`: Total number of parameters (baseline + covariates)

# Returns
NamedTuple with structure:
- `baseline`: NamedTuple of baseline parameters (shape=..., scale=...)
- `covariates`: NamedTuple of covariate coefficients (age=..., sex=...) [if present]

# Examples
```julia
# Weibull with covariates
params = build_hazard_params([log(1.5), log(0.2), 0.3, 0.1], 
                             [:shape, :scale, :age, :sex], 2, 4)
# Returns: (baseline = (shape = 0.405, scale = -1.609), 
#           covariates = (age = 0.3, sex = 0.1))

# Exponential without covariates  
params = build_hazard_params([log(0.5)], [:intercept], 1, 1)
# Returns: (baseline = (intercept = -0.693),)
```
"""
function build_hazard_params(
    log_scale_params::AbstractVector{T},
    parnames::Vector{Symbol},
    npar_baseline::Int,
    npar_total::Int
) where {T<:Real}
    
    # Validate inputs
    @assert length(log_scale_params) == npar_total "Parameter vector length mismatch"
    @assert length(parnames) == npar_total "Parameter names length mismatch"
    @assert npar_baseline <= npar_total "Baseline parameters exceed total"
    
    # Extract baseline parameter names and values
    baseline_names = parnames[1:npar_baseline]
    baseline_values = log_scale_params[1:npar_baseline]
    
    # Create NamedTuple for baseline
    baseline = NamedTuple{Tuple(baseline_names)}(baseline_values)
    
    # Handle covariates if present
    if npar_total > npar_baseline
        covar_names = parnames[(npar_baseline+1):npar_total]
        covar_values = log_scale_params[(npar_baseline+1):npar_total]
        covariates = NamedTuple{Tuple(covar_names)}(covar_values)
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

#### Update All Callers

**Caller 1: `rebuild_parameters` in `src/helpers.jl` line ~18**

Current:
```julia
hazard_pars = build_hazard_params(
    newvalues[i],
    model.hazards[i].npar_baseline,
    model.hazards[i].npar_total
)
```

Change to:
```julia
hazard_pars = build_hazard_params(
    newvalues[i],
    model.hazards[i].parnames,      # NEW: pass parameter names
    model.hazards[i].npar_baseline,
    model.hazards[i].npar_total
)
```

**Caller 2: `build_parameters` in `src/modelgeneration.jl` line ~761**

Current (approximate location):
```julia
for (hazname, hazind) in model.hazkeys
    hazard = model.hazards[hazind]
    initial_pars[hazname] = build_hazard_params(
        params_vec[hazind],
        hazard.npar_baseline,
        hazard.npar_total
    )
end
```

Change to:
```julia
for (hazname, hazind) in model.hazkeys
    hazard = model.hazards[hazind]
    initial_pars[hazname] = build_hazard_params(
        params_vec[hazind],
        hazard.parnames,                # NEW
        hazard.npar_baseline,
        hazard.npar_total
    )
end
```

#### Testing

**File: `test/test_helpers.jl` (add new tests)**

```julia
@testset "build_hazard_params - NamedTuple structure" begin
    # Test 1: Weibull baseline only
    params = build_hazard_params(
        [log(1.5), log(0.2)],
        [:shape, :scale],
        2, 2
    )
    
    @test params isa NamedTuple
    @test haskey(params, :baseline)
    @test params.baseline isa NamedTuple
    @test params.baseline.shape ≈ log(1.5)
    @test params.baseline.scale ≈ log(0.2)
    @test !haskey(params, :covariates)
    
    # Test 2: Weibull with covariates
    params = build_hazard_params(
        [log(1.5), log(0.2), 0.3, 0.1],
        [:shape, :scale, :age, :sex],
        2, 4
    )
    
    @test haskey(params, :baseline)
    @test haskey(params, :covariates)
    @test params.baseline.shape ≈ log(1.5)
    @test params.baseline.scale ≈ log(0.2)
    @test params.covariates.age ≈ 0.3
    @test params.covariates.sex ≈ 0.1
    
    # Test 3: Exponential baseline only
    params = build_hazard_params(
        [log(0.5)],
        [:intercept],
        1, 1
    )
    
    @test params.baseline.intercept ≈ log(0.5)
    
    # Test 4: Error on mismatched lengths
    @test_throws AssertionError build_hazard_params(
        [1.0, 2.0],
        [:shape],  # Wrong length!
        2, 2
    )
end
```

---


### Task 1.3: Update Helper Functions for NamedTuple Extraction

**Duration**: 2-3 hours

**Objective**: Provide functions to extract parameter values from NamedTuple structure for legacy code compatibility.

#### Changes Required

**File: `src/helpers.jl`**

**Location: Lines 285-340 - Parameter extraction functions**

Add new helper functions after line 283 (after `build_hazard_params`):

```julia
"""
    extract_baseline_values(hazard_params::NamedTuple)

Extract baseline parameter values as vector from NamedTuple structure.
"""
function extract_baseline_values(hazard_params::NamedTuple)
    return collect(values(hazard_params.baseline))
end

"""
    extract_covariate_values(hazard_params::NamedTuple)

Extract covariate coefficient values as vector from NamedTuple structure.
"""
function extract_covariate_values(hazard_params::NamedTuple)
    return haskey(hazard_params, :covariates) ? 
           collect(values(hazard_params.covariates)) : Float64[]
end

"""
    extract_params_vector(hazard_params::NamedTuple)

Extract full parameter vector (baseline + covariates) from NamedTuple structure.
"""
function extract_params_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_vals, covar_vals)
    else
        return baseline_vals
    end
end

"""
    extract_natural_vector(hazard_params::NamedTuple)

Extract parameter vector on natural scale for model.parameters.natural.
Applies exp() to baseline, keeps covariates as-is.
"""
function extract_natural_vector(hazard_params::NamedTuple)
    baseline_natural = exp.(collect(values(hazard_params.baseline)))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end
```

#### Testing

**File: `test/test_helpers.jl`**

```julia
@testset "Parameter extraction helpers" begin
    params_with_covars = (
        baseline = (shape = log(1.5), scale = log(0.2)),
        covariates = (age = 0.3, sex = 0.1)
    )
    
    # Test extract_baseline_values
    baseline_vals = extract_baseline_values(params_with_covars)
    @test baseline_vals ≈ [log(1.5), log(0.2)]
    
    # Test extract_covariate_values
    covar_vals = extract_covariate_values(params_with_covars)
    @test covar_vals ≈ [0.3, 0.1]
    
    # Test extract_params_vector
    all_params = extract_params_vector(params_with_covars)
    @test all_params ≈ [log(1.5), log(0.2), 0.3, 0.1]
    
    # Test extract_natural_vector
    natural_vals = extract_natural_vector(params_with_covars)
    @test natural_vals ≈ [1.5, 0.2, 0.3, 0.1]
end
```

---

### Task 1.4: Verify ParameterHandling.jl Compatibility

**Duration**: 1-2 hours

**Objective**: Confirm ParameterHandling.jl flatten/unflatten works with nested NamedTuples.

#### Testing

**File: `test/test_helpers.jl`**

```julia
@testset "ParameterHandling.jl with nested NamedTuples" begin
    using ParameterHandling
    
    # Test nested structure
    params = (
        h12 = (
            baseline = (shape = log(1.5), scale = log(0.2)),
            covariates = (age = 0.3, sex = 0.1)
        ),
        h23 = (
            baseline = (intercept = log(0.8),)
        )
    )
    
    # Flatten and unflatten
    flat, unflatten = ParameterHandling.flatten(params)
    reconstructed = unflatten(flat)
    
    # Verify structure preserved
    @test reconstructed.h12.baseline.shape ≈ log(1.5)
    @test reconstructed.h12.baseline.scale ≈ log(0.2)
    @test reconstructed.h12.covariates.age ≈ 0.3
    @test reconstructed.h23.baseline.intercept ≈ log(0.8)
    
    # Test modification (as in optimization)
    modified_flat = flat .+ 0.1
    modified = unflatten(modified_flat)
    @test modified.h12.baseline.shape ≈ log(1.5) + 0.1
end
```

---

## Phase 2: Update Hazard Functions for Named Access

**Goal**: Modify hazard generators to access parameters by name.

**Estimated Effort**: 15-20 hours

---

### Task 2.1: Update Linear Predictor Builder

**Duration**: 2-3 hours

**Objective**: Create function to build linear predictor expressions with named covariate access.

#### Changes Required

**File: `src/hazards.jl`**

**Location: After line 85 - Add new function**

```julia
"""
    _build_linear_pred_expr_named(covar_names::Vector{Symbol})

Build linear predictor expression that accesses covariates by name.

Generates: covars[1] * pars.covariates.age + covars[2] * pars.covariates.sex + ...

Note: Covariate VALUES (covars[i]) are positional (from data matrix).
      Covariate COEFFICIENTS (pars.covariates.XXX) are named.
"""
function _build_linear_pred_expr_named(covar_names::Vector{Symbol})
    if isempty(covar_names)
        return :(0.0)
    end
    
    # Build sum of covars[i] * pars.covariates.name
    terms = [:(covars[$i] * pars.covariates.$(covar_names[i])) 
             for i in eachindex(covar_names)]
    
    # Create addition expression
    if length(terms) == 1
        return terms[1]
    else
        return Expr(:call, :+, terms...)
    end
end
```

#### Testing

```julia
@testset "Linear predictor builder" begin
    # Test expression generation
    expr1 = _build_linear_pred_expr_named([:age])
    @test expr1 == :(covars[1] * pars.covariates.age)
    
    expr2 = _build_linear_pred_expr_named([:age, :sex])
    # Should generate sum of two terms
    
    expr_empty = _build_linear_pred_expr_named(Symbol[])
    @test expr_empty == :(0.0)
end
```

---

### Task 2.2: Update Exponential Hazard Generator

**Duration**: 2-3 hours

**File: `src/hazards.jl` Lines 348-380**

Current accesses `pars[1]` for intercept.

Change to `pars.baseline.intercept`.

See implementation details in full spec above.

---

### Task 2.3: Update Weibull Hazard Generator  

**Duration**: 3-4 hours

**File: `src/hazards.jl` Lines 395-450**

Current accesses `pars[1], pars[2]` for shape, scale.

Change to `pars.baseline.shape`, `pars.baseline.scale`.

Update both PH and AFT variants.

---

### Task 2.4: Update Gompertz Hazard Generator

**Duration**: 3-4 hours

**File: `src/hazards.jl` Lines 455-515**

Similar to Weibull - change to named baseline access.

---

### Task 2.5: Update Hazard Callers in Likelihood Functions

**Duration**: 4-6 hours

**File: `src/likelihoods.jl` Lines 1803-1920**

**Objective**: Pass NamedTuple parameters by hazard name to hazard functions.

**Challenge**: Current code uses hazard indices (1, 2, 3), need hazard names.

**Solution**: Add `hazard_names` vector to function signature.

```julia
function _compute_path_loglik_fused(
    path::SamplePath,
    pars,  # NamedTuple by hazard name (from unflatten)
    hazards::Vector{<:_Hazard},
    totalhazards::Vector{<:_TotalHazard},
    hazard_names::Vector{Symbol},  # NEW: maps index → name
    tmat::Matrix{Int64},
    # ... rest of parameters ...
)
    # Loop through total hazards
    for tothaz in totalhazards
        # Access parameters for each component hazard
        for h in tothaz.components
            hazname = hazard_names[h]  # Convert index to name
            hazard_pars = pars[hazname]  # Get (baseline=..., covariates=...)
            
            # Call hazard function with named parameters
            haz_value = eval_hazard(hazards[h], ub, hazard_pars, covars)
            # ... rest of computation ...
        end
    end
end
```

**Update all callers** (lines 700-750) to pass hazard_names:
```julia
hazard_names = collect(keys(model.parameters.nested))
loglik = _compute_path_loglik_fused(path, pars, hazards, totalhazards, 
                                    hazard_names, tmat, ...)
```

---

### Task 2.6: Update Simulation Code

**Duration**: 2-3 hours

**File: `src/simulation.jl` Lines 593-650**

Similar updates - pass NamedTuple parameters by name when calling hazard functions.

---

### Phase 2 Testing Strategy

**Integration Tests**: Build models and verify:
1. Hazard evaluation produces same results as before
2. Likelihood computation unchanged
3. Simulation produces valid paths
4. Parameter optimization still works

**Performance Benchmarks**:
```julia
@benchmark loglik_exact($model, $data)  # Target: <5% overhead
@benchmark simulate($model)              # Target: <5% overhead
```

---

## Phase 3: Validated Parameter Setting (Next Document Section)

This completes Phase 1 and Phase 2 detailed specifications. Phase 3 will cover validation logic for `set_parameters!`.

