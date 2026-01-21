# Phase-Type Surrogate Fitting Implementation Plan

**Created**: 2026-01-18  
**Status**: Ready for implementation  
**Branch**: `penalized_splines`

---

## Executive Summary

The phase-type surrogate is currently **only initialized from Markov rates** (`_build_phasetype_from_markov`). The `ProposalConfig.optimize` field exists but is **never used**. This plan outlines the implementation of actual MLE fitting for phase-type surrogates.

Additionally, time-varying covariates (TVC) are not properly handled during path sampling and likelihood computation—this must be fixed first as a blocking prerequisite.

---

## User Decisions (2026-01-18)

| Question | Decision |
|----------|----------|
| TVC handling | **Blocking prerequisite** - Phase 0 must complete before fitting |
| Multi-source fitting | **Joint optimization** - Single objective over all phase-type source states |
| Refitting during MCEM | **No** - One-time fit at surrogate initialization |
| Default constraint | **`:sctp`** - No eigenvalue ordering unless user specifies `:ordered_sctp` |
| Covariate effects | **`:homogeneous`** - Current spec (covariates on exit rates only) |

---

## Phase 0: Fix TVC Handling (BLOCKING PREREQUISITE)

### Problem Analysis

| Location | Issue |
|----------|-------|
| `src/inference/sampling_markov.jl` L147 | Uses single covariate index for entire subject |
| `src/inference/sampling_phasetype.jl` L215-396 | `convert_expanded_path_to_censored_data` doesn't include covariates or per-interval Q matrices |
| `src/inference/sampling_phasetype.jl` L382 | `tpm_book = [tpms]` - single covariate level |

### Task 0.1: Add Covariate Support to Path Conversion

**File**: `src/inference/sampling_phasetype.jl`

**Changes to `convert_expanded_path_to_censored_data`**:

1. **Add new parameters**:
```julia
function convert_expanded_path_to_censored_data(
    expanded_path::SamplePath,
    surrogate::PhaseTypeSurrogate,
    original_subj_data::SubDataFrame;  # NEW: Original subject data with covariates
    hazmat_book::Dict{Int, Matrix{Float64}},  # NEW: Covariate-indexed Q matrices
    schur_cache::Union{Nothing, Dict{Int, CachedSchurDecomposition}} = nothing  # NEW: Dict
)
```

2. **Interpolate covariates at transition times**:
   - For each interval `[t_start, t_trans]` in the sampled path
   - Find original data row where `tstart ≤ t_start < tstop`
   - Copy covariate values from that row
   - Use existing pattern from `_interpolate_covariates_from_paths!` in `src/utilities/initialization.jl`

3. **Build covariate-indexed tpm_book**:
   - Extract unique covariate combinations present in the path
   - For each unique combo, compute Q matrix lookup and TPMs
   - `tpm_map[i, 1]` = covariate combo index for interval i
   - `tpm_map[i, 2]` = time index within that combo

4. **Add covariate columns to output DataFrame**

### Task 0.2: Update Calling Code in `sampling_markov.jl`

**File**: `src/inference/sampling_markov.jl` (around L183-190)

**Changes**:
1. Pass `subj_data = view(model.data, subj_inds, :)` to `convert_expanded_path_to_censored_data`
2. Pass full `hazmat_book_ph` dictionary (not single entry)
3. Pass full `schur_cache_ph` dictionary (not single entry)

### Task 0.3: Verify Forward Likelihood Handles Multi-Covariate

**File**: `src/inference/sampling_phasetype.jl`

The `compute_forward_loglik` function should already handle multi-covariate `tpm_book` correctly since it indexes via `tpm_map`. Verify:
- `tpm_book[covar_idx][time_idx]` is used correctly
- `hazmat_book[covar_idx]` is passed for exact observations

### Task 0.4: Add TVC Validation Test

**File**: `MultistateModelsTests/unit/test_phasetype_tvc.jl` (new)

Test case:
1. Create illness-death model with time-varying treatment covariate
2. Build phase-type surrogate  
3. Sample paths for subject with TVC
4. Verify surrogate likelihood uses different Q matrices for different intervals

---

## Phase 1: Parameterization Infrastructure

### Task 1.1: Define `PhaseTypeParameters` Struct

**File**: `src/phasetype/types.jl`

```julia
"""
Parameters for a phase-type distribution under SCTP constraints.

For an n-phase Coxian with K destinations:
- `λ`: n-1 progression rates (phase 1→2→...→n), always positive
- `μ`: n×K exit rates to each destination, always non-negative
- `n_phases`: Number of phases
- `destinations`: Vector of destination state indices
- `source_state`: The observed source state this parameterizes
"""
struct PhaseTypeParameters
    λ::Vector{Float64}      # Progression rates (length n-1)
    μ::Matrix{Float64}      # Exit rates (phases × destinations)
    n_phases::Int
    destinations::Vector{Int}
    source_state::Int
end
```

### Task 1.2: Implement Parameter Extraction

**File**: `src/phasetype/surrogate.jl`

```julia
"""
    extract_phasetype_parameters(Q::Matrix, surrogate::PhaseTypeSurrogate, source_state::Int)

Extract PhaseTypeParameters from Q matrix for transitions out of source_state.
"""
function extract_phasetype_parameters(Q::Matrix, surrogate::PhaseTypeSurrogate, source_state::Int)
    phases = surrogate.state_to_phases[source_state]
    n = length(phases)
    
    # Find destination states (those with non-zero rates from any phase)
    destinations = Int[]
    for k in 1:surrogate.n_observed_states
        if k != source_state
            dest_phases = surrogate.state_to_phases[k]
            if any(Q[phases, dest_phases[1]] .> 0)
                push!(destinations, k)
            end
        end
    end
    
    # Extract progression rates (sub-diagonal of phase block)
    λ = [Q[phases[i], phases[i+1]] for i in 1:(n-1)]
    
    # Extract exit rates to each destination
    μ = zeros(n, length(destinations))
    for (k_idx, k) in enumerate(destinations)
        dest_phase = surrogate.state_to_phases[k][1]  # First phase of destination
        μ[:, k_idx] = [Q[phases[i], dest_phase] for i in 1:n]
    end
    
    return PhaseTypeParameters(λ, μ, n, destinations, source_state)
end
```

### Task 1.3: Implement Parameter Injection

**File**: `src/phasetype/surrogate.jl`

```julia
"""
    inject_phasetype_parameters!(Q::Matrix, params::PhaseTypeParameters, 
                                  surrogate::PhaseTypeSurrogate)

Inject PhaseTypeParameters back into Q matrix, updating diagonal entries.
"""
function inject_phasetype_parameters!(Q::Matrix, params::PhaseTypeParameters,
                                       surrogate::PhaseTypeSurrogate)
    phases = surrogate.state_to_phases[params.source_state]
    n = params.n_phases
    
    # Set progression rates
    for i in 1:(n-1)
        Q[phases[i], phases[i+1]] = params.λ[i]
    end
    
    # Set exit rates
    for (k_idx, k) in enumerate(params.destinations)
        dest_phase = surrogate.state_to_phases[k][1]
        for i in 1:n
            Q[phases[i], dest_phase] = params.μ[i, k_idx]
        end
    end
    
    # Update diagonal (row sums must be zero)
    for i in 1:n
        Q[phases[i], phases[i]] = -sum(Q[phases[i], j] for j in 1:size(Q,2) if j != phases[i])
    end
    
    return Q
end
```

### Task 1.4: Implement Flat ↔ Structured Conversion (Joint Over All Sources)

**File**: `src/phasetype/surrogate.jl`

```julia
"""
    flatten_all_phasetype_parameters(params_list::Vector{PhaseTypeParameters}) -> Vector{Float64}

Convert all source-state parameters to single flat optimization vector.
All parameters in log-scale for unconstrained optimization.
"""
function flatten_all_phasetype_parameters(params_list::Vector{PhaseTypeParameters})
    x = Float64[]
    for params in params_list
        # λ values (length n-1)
        append!(x, log.(params.λ))
        # μ values (n × K, column-major), with small offset for zeros
        append!(x, vec(log.(params.μ .+ 1e-12)))
    end
    return x
end

"""
    unflatten_all_phasetype_parameters(x::Vector, templates::Vector{PhaseTypeParameters})

Convert flat vector back to structured parameters for all source states.
"""
function unflatten_all_phasetype_parameters(x::Vector, templates::Vector{PhaseTypeParameters})
    result = PhaseTypeParameters[]
    offset = 0
    for template in templates
        n = template.n_phases
        K = length(template.destinations)
        
        n_λ = n - 1
        n_μ = n * K
        
        λ = exp.(x[offset+1 : offset+n_λ])
        μ = reshape(exp.(x[offset+n_λ+1 : offset+n_λ+n_μ]), n, K)
        
        push!(result, PhaseTypeParameters(λ, μ, n, template.destinations, template.source_state))
        offset += n_λ + n_μ
    end
    return result
end
```

---

## Phase 2: SCTP Constraint Functions

### Task 2.1: Soft Penalty (Default - No Ordering)

**File**: `src/phasetype/constraints.jl` (new file)

```julia
"""
SCTP (Sufficient Conditions for Total Positivity) constraints for phase-type distributions.

For an n-phase Coxian with exit rates μ from each phase:
1. λᵢ > 0 for i = 1, ..., n-1 (progression rates positive) - enforced by log-scale
2. μᵢ ≥ 0 for i = 1, ..., n (exit rates non-negative) - enforced by log-scale
3. λᵢ ≠ λⱼ for i ≠ j (distinct eigenvalues) - soft penalty

Note: Eigenvalue ORDERING (λ₁ < λ₂ < ... < λₙ) is NOT enforced by default.
Use :ordered_sctp if strict ordering is desired.
"""

"""
    sctp_penalty(params_list::Vector{PhaseTypeParameters}; min_separation::Float64 = 0.01)

Soft penalty for eigenvalue separation across all source states.
Returns 0 if well-separated, positive penalty otherwise.
"""
function sctp_penalty(params_list::Vector{PhaseTypeParameters}; min_separation::Float64 = 0.01)
    penalty = 0.0
    for params in params_list
        λ = params.λ
        n = length(λ)
        for i in 1:n
            for j in (i+1):n
                gap = abs(λ[i] - λ[j])
                if gap < min_separation
                    penalty += (min_separation - gap)^2 / min_separation^2
                end
            end
        end
    end
    return penalty
end

"""
    get_all_parameter_bounds(templates::Vector{PhaseTypeParameters};
                              rate_lb::Float64 = 1e-6, rate_ub::Float64 = 1e3)

Get lower and upper bounds for joint optimization (log-scale).
"""
function get_all_parameter_bounds(templates::Vector{PhaseTypeParameters};
                                   rate_lb::Float64 = 1e-6, rate_ub::Float64 = 1e3)
    n_params = sum(t -> (t.n_phases - 1) + t.n_phases * length(t.destinations), templates)
    lb = fill(log(rate_lb), n_params)
    ub = fill(log(rate_ub), n_params)
    return lb, ub
end
```

---

## Phase 3: Optimization Objective (Joint)

### Task 3.1: Joint Likelihood Function

**File**: `src/phasetype/fitting.jl` (new file)

```julia
"""
    phasetype_total_loglik(x::Vector{Float64}, surrogate::PhaseTypeSurrogate,
                           model::MultistateProcess, templates::Vector{PhaseTypeParameters};
                           sctp_weight::Float64 = 10.0)

Total log-likelihood across all subjects for joint optimization over all source states.

# Arguments
- `x`: Flat parameter vector (log-scale) for ALL source states
- `surrogate`: PhaseTypeSurrogate with current Q matrix
- `model`: MultistateProcess with data
- `templates`: Template parameters for each source state (for unflattening)
- `sctp_weight`: Weight on SCTP penalty

# Returns
- Negative total log-likelihood (for minimization)
"""
function phasetype_total_loglik(x::Vector{Float64}, surrogate::PhaseTypeSurrogate,
                                 model::MultistateProcess, 
                                 templates::Vector{PhaseTypeParameters};
                                 sctp_weight::Float64 = 10.0)
    # Unflatten parameters for all source states
    params_list = unflatten_all_phasetype_parameters(x, templates)
    
    # Build Q matrix with new parameters
    Q = copy(surrogate.expanded_Q)
    for params in params_list
        inject_phasetype_parameters!(Q, params, surrogate)
    end
    
    # Compute Schur decomposition for efficient TPM computation
    schur_cache = CachedSchurDecomposition(Q)
    
    # Build TPM books for all unique time intervals
    tpm_book, tpm_map = build_phasetype_tpm_book_from_Q(Q, model, surrogate, schur_cache)
    emat = build_phase_emission_matrix(model.data, surrogate)
    
    # Sum log-likelihood across subjects
    total_ll = 0.0
    for subj in 1:length(model.subjectindices)
        subj_inds = model.subjectindices[subj]
        subj_data = view(model.data, subj_inds, :)
        subj_emat = view(emat, subj_inds, :)
        subj_tpm_map = view(tpm_map, subj_inds, :)
        
        ll = compute_forward_loglik(subj_data, subj_emat, subj_tpm_map, tpm_book, Q,
                                    surrogate.n_expanded_states)
        total_ll += ll
    end
    
    # Add SCTP penalty
    penalty = sctp_weight * sctp_penalty(params_list)
    
    return -(total_ll - penalty)  # Return negative for minimization
end
```

---

## Phase 4: Main Fitting Function

### Task 4.1: Main Entry Point (One-Time Fit)

**File**: `src/phasetype/fitting.jl`

```julia
"""
    fit_phasetype_surrogate!(surrogate::PhaseTypeSurrogate, model::MultistateProcess;
                              maxiter::Int = 100,
                              gtol::Float64 = 1e-4,
                              sctp_weight::Float64 = 10.0,
                              verbose::Bool = false)

Fit phase-type surrogate parameters to maximize likelihood.

Uses L-BFGS optimization with box constraints. All source states with
multi-phase expansions are fit jointly in a single optimization.

# Arguments
- `surrogate::PhaseTypeSurrogate`: Surrogate to fit (modified in place)
- `model::MultistateProcess`: Model with data
- `maxiter::Int`: Maximum iterations
- `gtol::Float64`: Gradient tolerance for convergence
- `sctp_weight::Float64`: Weight on SCTP eigenvalue separation penalty
- `verbose::Bool`: Print optimization progress

# Returns
- NamedTuple with: converged, iterations, final_objective, parameters
"""
function fit_phasetype_surrogate!(surrogate::PhaseTypeSurrogate, 
                                   model::MultistateProcess;
                                   maxiter::Int = 100,
                                   gtol::Float64 = 1e-4,
                                   sctp_weight::Float64 = 10.0,
                                   verbose::Bool = false)
    # Find all source states with multi-phase expansions
    source_states = [s for s in 1:surrogate.n_observed_states 
                     if length(surrogate.state_to_phases[s]) > 1]
    
    if isempty(source_states)
        @warn "No multi-phase states in surrogate; nothing to fit"
        return (converged = true, iterations = 0, final_objective = 0.0, parameters = nothing)
    end
    
    # Extract current parameters for all source states
    templates = [extract_phasetype_parameters(surrogate.expanded_Q, surrogate, s) 
                 for s in source_states]
    
    # Get initial parameters and bounds
    x0 = flatten_all_phasetype_parameters(templates)
    lb, ub = get_all_parameter_bounds(templates)
    
    # Define objective function
    function objective(x, p)
        return phasetype_total_loglik(x, surrogate, model, templates; sctp_weight=sctp_weight)
    end
    
    # Set up Optimization.jl problem
    opt_f = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(opt_f, x0; lb=lb, ub=ub)
    
    # Solve using L-BFGS-B
    sol = solve(prob, Optim.LBFGS(); 
                maxiters=maxiter, 
                g_tol=gtol,
                show_trace=verbose)
    
    # Update surrogate with fitted parameters
    fitted_params_list = unflatten_all_phasetype_parameters(sol.u, templates)
    for params in fitted_params_list
        inject_phasetype_parameters!(surrogate.expanded_Q, params, surrogate)
    end
    
    return (
        converged = sol.retcode == ReturnCode.Success,
        iterations = sol.stats.iterations,
        final_objective = sol.objective,
        parameters = fitted_params_list
    )
end
```

---

## Phase 5: Integration (One-Time at Surrogate Build)

### Task 5.1: Update `_build_phasetype_from_markov`

**File**: `src/surrogate/markov.jl`

Add fitting call after initialization:

```julia
# At end of _build_phasetype_from_markov, before return:
if config.optimize
    fit_result = fit_phasetype_surrogate!(phasetype_surrogate, model;
                                          maxiter = config.maxiter,
                                          gtol = config.gtol,
                                          verbose = config.verbose)
    if !fit_result.converged && config.verbose
        @warn "Phase-type surrogate fitting did not converge" fit_result.iterations
    end
end
```

### Task 5.2: NO Changes to MCEM Loop

Per user decision: no refitting during MCEM iterations. The `refit_frequency` field in `ProposalConfig` is not needed.

---

## Phase 6: Testing

### Unit Tests

| Test File | Purpose |
|-----------|---------|
| `test_phasetype_params.jl` | Parameter extract/inject round-trip |
| `test_phasetype_constraints.jl` | SCTP penalty computation |
| `test_phasetype_tvc.jl` | TVC handling (Phase 0) |

### Integration Tests

| Test File | Purpose |
|-----------|---------|
| `test_phasetype_fitting.jl` | End-to-end fitting, likelihood improvement |

### Long Tests

| Test File | Purpose |
|-----------|---------|
| `longtest_phasetype_fitting.jl` | Parameter recovery with simulated data |

---

## Files Summary

| File | Type | Description |
|------|------|-------------|
| `src/phasetype/types.jl` | Modify | Add `PhaseTypeParameters` struct |
| `src/phasetype/constraints.jl` | **New** | SCTP penalty and bounds |
| `src/phasetype/fitting.jl` | **New** | Main fitting functions |
| `src/phasetype/surrogate.jl` | Modify | Add extract/inject/flatten functions |
| `src/inference/sampling_phasetype.jl` | Modify | TVC fixes (Phase 0) |
| `src/inference/sampling_markov.jl` | Modify | TVC fixes (Phase 0) |
| `src/surrogate/markov.jl` | Modify | Call fitting after initialization |

---

## Implementation Order

1. **Phase 0** - TVC bug fixes (blocking)
2. **Phase 1** - Parameterization infrastructure
3. **Phase 2** - SCTP constraints
4. **Phase 3** - Optimization objective
5. **Phase 4** - Main fitting function
6. **Phase 5** - Integration
7. **Phase 6** - Testing

Estimated effort: 2-3 days of focused implementation
