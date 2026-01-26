# Penalized Likelihood Refactoring Plan: Markov Panel and MCEM

**Document Version**: 1.1  
**Created**: 2026-01-24  
**Last Updated**: 2026-01-24  
**Status**: ✅ ALL PHASES COMPLETE  
**Branch**: `penalized_splines`  
**Prerequisite**: Exact data refactoring complete (all 5 phases)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Completed Infrastructure (From Exact Data Refactoring)](#2-completed-infrastructure)
3. [Current State Analysis](#3-current-state-analysis)
4. [Design Principles](#4-design-principles)
5. [Markov Panel Specifications](#5-markov-panel-specifications)
6. [MCEM Specifications](#6-mcem-specifications)
7. [File Changes](#7-file-changes)
8. [Implementation Phases](#8-implementation-phases)
9. [Testing Requirements](#9-testing-requirements)
10. [Agent Handoff Protocol](#10-agent-handoff-protocol)

---

## 1. Executive Summary

### Goal
Extend the refactored penalized likelihood infrastructure to support:
- **Markov panel data** (`_fit_markov_panel`) via matrix exponential likelihood
- **Semi-Markov MCEM** (`_fit_mcem`) via Monte Carlo EM with importance sampling

### Key Insight
The exact data refactoring (Phases 1-5) built a complete infrastructure that can be **reused**:
- Type hierarchy: `AbstractPenalty`, `AbstractHyperparameterSelector`, `HyperparameterSelectionResult`
- Interface functions: `compute_penalty`, `n_hyperparameters`, `get_hyperparameters`, etc.
- Helper functions: `_resolve_penalty`, `_resolve_selector`, `build_penalty_config`
- Selection infrastructure: `_select_hyperparameters`, `_fit_inner_coefficients`

### What's New (This Plan)
1. **Markov panel**: Create `_fit_markov_panel_penalized` with λ selection
2. **MCEM**: Refactor existing penalty handling to use new architecture with λ selection

### Complexity Comparison

| Component | Exact Data | Markov Panel | MCEM |
|-----------|-----------|--------------|------|
| Likelihood | `loglik_exact` | `loglik_markov` | `loglik_semi_markov` |
| Optimization | Single MLE | Single MLE | Iterative EM |
| λ Selection | Nested optimization | Nested optimization | **Nested within EM** |
| Key Challenge | None | AD backend selection | MC variance in λ selection |

---

## 2. Completed Infrastructure (From Exact Data Refactoring)

### 2.1 Type System (Fully Reusable)

**File**: `src/types/penalties.jl`

```julia
# Abstract types
AbstractPenalty                      # Base for penalty types
AbstractHyperparameterSelector       # Base for selection strategies

# Penalty types
NoPenalty                            # Unpenalized MLE
QuadraticPenalty                     # P(β;λ) = (1/2)Σⱼ λⱼ βⱼᵀSⱼβⱼ
PenaltyConfig = QuadraticPenalty     # Backward compatibility alias

# Selector types
NoSelection                          # Fixed λ
PIJCVSelector(nfolds)                # Newton-approximated CV
ExactCVSelector(nfolds)              # Exact k-fold CV
REMLSelector()                       # REML/EFS criterion
PERFSelector()                       # PERF criterion

# Result type
HyperparameterSelectionResult        # Returned by selection functions
```

### 2.2 Interface Functions (Fully Reusable)

```julia
# Penalty interface (penalties.jl)
compute_penalty(params, penalty)      # → penalty contribution
n_hyperparameters(penalty)            # → number of λ values
get_hyperparameters(penalty)          # → current λ vector
set_hyperparameters(penalty, λ)       # → new penalty with updated λ
hyperparameter_bounds(penalty)        # → (lb, ub) for optimization
has_penalties(penalty)                # → Bool

# Helpers (fit_common.jl, penalty_config.jl)
_resolve_penalty(penalty, model)      # Symbol → SplinePenalty or nothing
_resolve_selector(symbol, penalty)    # Symbol → AbstractHyperparameterSelector
build_penalty_config(model, specs)    # SplinePenalty → QuadraticPenalty
```

### 2.3 Selection Infrastructure (Requires Data-Type Specialization)

**File**: `src/inference/smoothing_selection.jl`

```julia
# Main dispatcher - NEEDS SPECIALIZATION
_select_hyperparameters(model, data, penalty, selector; kwargs...)
    # Currently only handles ExactData
    # Needs overloads for MPanelData and MCEM

# Inner optimization - NEEDS SPECIALIZATION  
_fit_inner_coefficients(model, data, penalty, beta_init; kwargs...)
    # Currently calls loglik_exact
    # Needs overloads for loglik_markov and loglik_semi_markov

# Nested optimization functions - REUSABLE
_nested_optimization_pijcv(...)       # Wood 2024 NCV
_grid_search_exact_cv(...)            # Grid search CV
_nested_optimization_reml(...)        # REML criterion
_nested_optimization_perf(...)        # PERF criterion
```

### 2.4 Final Fitting (Requires Data-Type Specialization)

**File**: `src/inference/fit_penalized.jl`

```julia
# Currently handles exact data only
_fit_exact_penalized(model, data, paths, penalty, selector; kwargs...)
_fit_coefficients_at_fixed_hyperparameters(model, data, penalty, beta; kwargs...)
    # Uses loglik_exact - needs overloads for other data types
```

---

## 3. Current State Analysis

### 3.1 Markov Panel (`_fit_markov_panel`)

**File**: `src/inference/fit_markov.jl` (~254 lines)

**Current Behavior**:
- No penalty support at all
- Uses `loglik_markov` with ForwardDiff by default
- Returns `nothing` for `smoothing_parameters` and `edf` fields
- No λ selection capability

**Key Code Path**:
```julia
function _fit_markov_panel(model; constraints=nothing, verbose=true, 
                           solver=nothing, adbackend=ForwardDiffBackend(), 
                           vcov_type=:ij, ...)
    books = build_tpm_mapping(model.data)
    parameters = get_parameters_flat(model)
    lb, ub = model.bounds.lb, model.bounds.ub
    
    loglik_fn = (p, d) -> loglik_markov(p, d; backend=adbackend)
    optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend))
    prob = OptimizationProblem(optf, parameters, MPanelData(model, books); lb=lb, ub=ub)
    
    sol = _solve_optimization(prob, solver)
    vcov, vcov_type_used, subject_grads = _compute_vcov_markov(sol.u, model, books, vcov_type; ...)
    
    return MultistateModelFitted(...)
end
```

**Required Changes**:
1. Add penalty resolution (`_resolve_penalty`, `build_penalty_config`)
2. Add selector resolution (`_resolve_selector`)
3. Dispatch to `_fit_markov_panel_penalized` when penalties active
4. Create penalized likelihood function: `loglik_markov_penalized`
5. Create Markov-specific inner coefficient fitting

### 3.2 MCEM (`_fit_mcem`)

**File**: `src/inference/fit_mcem.jl` (~1300 lines)

**Current Behavior**:
- Has basic penalty support via `penalty=:auto` kwarg
- Creates `penalty_config` and uses `loglik_semi_markov_penalized` 
- Uses **fixed λ only** - no automatic selection
- Penalty applied in M-step optimization

**Key Code Path (Penalty Handling)**:
```julia
function _fit_mcem(model; penalty=:auto, lambda_init=1.0, ...)
    resolved_penalty = _resolve_penalty(penalty, model)
    penalty_config = build_penalty_config(model, resolved_penalty; lambda_init=lambda_init)
    use_penalty = has_penalties(penalty_config)
    
    # In M-step optimization loop:
    if use_penalty
        penalized_loglik = (params, data) -> loglik(params, data, penalty_config)
        optf = OptimizationFunction(penalized_loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, params_cur, SMPanelData(...); lb=lb, ub=ub)
    else
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, params_cur, SMPanelData(...); lb=lb, ub=ub)
    end
    sol = _solve_optimization(prob, solver)
```

**Current Limitations**:
1. No λ selection - only uses `lambda_init`
2. λ is fixed throughout all MCEM iterations
3. No EDF or smoothing diagnostics returned

**Required Changes**:
1. Add `select_lambda` kwarg to `_fit_mcem`
2. Create MCEM-specific λ selection (at convergence or periodically)
3. Update penalty during MCEM iterations (optional, complex)
4. Return `smoothing_parameters` and `edf` in fitted model

---

## 4. Design Principles

### 4.1 Ipopt + ForwardDiff Everywhere (Inherited)

**HARD REQUIREMENT**: All optimizations use Ipopt with ForwardDiff.

This applies to:
- Markov panel MLE
- MCEM M-step optimization
- Inner loop coefficient fitting for λ selection
- Outer loop λ optimization

### 4.2 Reuse Over Reimplementation

Maximize reuse of existing infrastructure:
- Use existing type hierarchy unchanged
- Use existing `_select_hyperparameters` with dispatch specialization
- Use existing nested optimization functions unchanged

### 4.3 MCEM λ Selection Strategy

**Key Decision**: When to select λ in MCEM?

**CRITICAL INSIGHT**: Option A (select once at convergence) is INVALID because paths 
sampled at β̂(λ_init) are not valid for evaluating Q(β̂(λ_trial)) at different λ values.
The importance weights correct for the surrogate→target difference at FIXED β, not for
the change in β when λ changes.

**Option A: Select once at MCEM convergence** ~~(RECOMMENDED for Phase 1)~~ **INVALID - DO NOT USE**
```
MCEM iterations until convergence (fixed λ = lambda_init)
    ↓
At convergence: Select optimal λ via PIJCV  ← WRONG: paths from β̂(λ_init) invalid for β̂(λ_trial)
    ↓
Final M-step at selected λ
```

**Why Option A Fails**: When evaluating PIJCV at a trial λ_trial ≠ λ_init:
1. Inner fit produces β̂(λ_trial) ≠ β̂(λ_init)
2. The paths were sampled to be representative of β̂(λ_init), not β̂(λ_trial)
3. Importance weights don't correct for this - Q(β̂(λ_trial); paths, weights) is biased

**Option B: Select λ within each MCEM iteration** (CORRECT APPROACH)
```
for iter in 1:max_iterations
    E-step: Sample paths at β_current
    λ-step: Select λ to minimize PIJCV using current paths/weights  ← Valid: paths fixed
    M-step: Fit β at selected λ using current paths/weights
    Check convergence
end
```

**Why Option B is Correct**: Within each MCEM iteration:
- Paths/weights are FIXED for the entire iteration (sampled at β_current)
- λ selection uses the same paths/weights as the M-step
- Evaluating Q(β; paths, weights) is valid for ANY β (that's what importance weighting does)
- λ selection just finds the best regularization for the current MC approximation to Q
- Next E-step samples new paths appropriate for the updated β

**Pros**: Statistically valid; λ adapts during MCEM; uses same paths for selection and M-step
**Cons**: More computation per iteration (λ selection + M-step)

### 4.4 Markov Panel AD Backend Handling

Markov panel supports multiple AD backends:
- ForwardDiff (default, mutating implementation)
- Enzyme/Mooncake (functional implementation)

For penalized fitting with λ selection, we use ForwardDiff only:
- Inner loop optimization requires second-order AD
- Ipopt + ForwardDiff is our standard stack

---

## 5. Markov Panel Specifications

### 5.1 New Function: `_fit_markov_panel_penalized`

**File**: `src/inference/fit_markov.jl` (or new `fit_markov_penalized.jl`)

**Signature**:
```julia
function _fit_markov_panel_penalized(
    model::MultistateModel,
    books::Tuple,                      # TPM bookkeeping from build_tpm_mapping
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    constraints = nothing,
    verbose::Bool = true,
    solver = nothing,
    vcov_type::Symbol = :ij,
    vcov_threshold::Bool = true,
    loo_method::Symbol = :direct,
    inner_maxiter::Int = 100,
    lambda_init::Float64 = 1.0,
    kwargs...
) -> MultistateModelFitted
```

**Pseudocode**:
```julia
function _fit_markov_panel_penalized(model, books, penalty, selector; kwargs...)
    # 1. Get initial parameters and bounds
    parameters = get_parameters_flat(model)
    lb, ub = model.bounds.lb, model.bounds.ub
    
    # 2. Validate constraints
    if !isnothing(constraints) && !(selector isa NoSelection)
        throw(ArgumentError("Constraints with λ selection not supported"))
    end
    
    # 3. Create MPanelData container
    data = MPanelData(model, books)
    
    # 4. Select hyperparameters (if not NoSelection)
    smoothing_result = nothing
    final_penalty = penalty
    warmstart_beta = parameters
    
    if !(selector isa NoSelection)
        smoothing_result = _select_hyperparameters_markov(
            model, data, penalty, selector;
            beta_init=parameters, inner_maxiter=inner_maxiter, verbose=verbose
        )
        final_penalty = smoothing_result.penalty
        warmstart_beta = smoothing_result.warmstart_beta
    end
    
    # 5. Final fit at fixed λ
    sol = _fit_coefficients_at_fixed_hyperparameters_markov(
        model, data, final_penalty, warmstart_beta;
        lb=lb, ub=ub, solver=solver, maxiter=500, verbose=verbose
    )
    
    # 6. Warn about model-based variance with penalties
    if vcov_type == :model && has_penalties(final_penalty)
        @warn "Model-based variance may be inappropriate for penalized models..."
    end
    
    # 7. Compute variance
    vcov, vcov_type_used, subject_grads = _compute_vcov_markov(
        sol.u, model, books, vcov_type; ...
    )
    
    # 8. Assemble and return MultistateModelFitted
    return MultistateModelFitted(
        ...,
        smoothing_parameters = selected_lambda,
        edf = selected_edf
    )
end
```

### 5.2 New Function: `loglik_markov_penalized`

**File**: `src/likelihood/loglik_markov.jl`

**Signature**:
```julia
function loglik_markov_penalized(
    parameters, 
    data::MPanelData, 
    penalty::AbstractPenalty;
    neg::Bool = true,
    return_ll_subj::Bool = false
)
```

**Implementation**:
```julia
function loglik_markov_penalized(parameters, data::MPanelData, penalty::AbstractPenalty;
                                  neg::Bool=true, return_ll_subj::Bool=false)
    # Base Markov likelihood
    ll_base = _loglik_markov_mutating(parameters, data; neg=neg, return_ll_subj=false)
    
    # Add penalty (in negative log-likelihood convention)
    if neg
        return ll_base + compute_penalty(parameters, penalty)
    else
        return ll_base - compute_penalty(parameters, penalty)
    end
end

# Dispatch method for consistent interface
function loglik(parameters, data::MPanelData, penalty::AbstractPenalty; kwargs...)
    loglik_markov_penalized(parameters, data, penalty; kwargs...)
end
```

### 5.3 New Function: `_select_hyperparameters_markov`

**File**: `src/inference/smoothing_selection.jl` (add as method)

**Signature**:
```julia
function _select_hyperparameters(
    model::MultistateProcess,
    data::MPanelData,           # <-- Dispatch on MPanelData
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
) -> HyperparameterSelectionResult
```

**Implementation Strategy**:
The existing `_select_hyperparameters` dispatches on selector type. We need a **data-type** dispatch layer:

```julia
# Add to smoothing_selection.jl

# Markov panel dispatch (new)
function _select_hyperparameters(model::MultistateProcess, data::MPanelData, 
                                  penalty::AbstractPenalty, selector::AbstractHyperparameterSelector;
                                  kwargs...)
    # NoSelection: immediate return
    if selector isa NoSelection
        lambda = get_hyperparameters(penalty)
        edf = compute_edf_markov(beta_init, lambda, penalty, model, data)
        return HyperparameterSelectionResult(lambda, beta_init, penalty, NaN, edf, true, :none, 0, (;))
    end
    
    # Other selectors: delegate to nested optimization
    return _select_hyperparameters_impl(model, data, penalty, selector; kwargs...)
end
```

### 5.4 New Function: `_fit_inner_coefficients_markov`

**File**: `src/inference/smoothing_selection.jl`

```julia
function _fit_inner_coefficients(
    model::MultistateProcess,
    data::MPanelData,           # <-- Dispatch on MPanelData
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int = 50
) -> Vector{Float64}
    
    # Define penalized negative log-likelihood
    function penalized_nll(β, p)
        nll = loglik_markov(β, data; neg=true)
        pen = compute_penalty(β, penalty)
        return nll + pen
    end
    
    # Set up Ipopt optimization
    adtype = SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=lb, ub=ub)
    
    # Solve with relaxed tolerance for inner loop
    sol = solve(prob, IpoptOptimizer(additional_options=Dict("sb"=>"yes"));
                maxiters=maxiter, tol=LAMBDA_SELECTION_INNER_TOL, print_level=0)
    
    return sol.u
end
```

### 5.5 Updated `_fit_markov_panel` (Dispatcher)

**File**: `src/inference/fit_markov.jl`

Update existing function to be a dispatcher:

```julia
function _fit_markov_panel(model::MultistateModel; 
                           constraints = nothing, verbose = true, 
                           solver = nothing, adbackend::ADBackend = ForwardDiffBackend(), 
                           vcov_type::Symbol = :ij, vcov_threshold = true,
                           loo_method = :direct,
                           # NEW penalty kwargs:
                           penalty = :auto, lambda_init::Float64 = 1.0, 
                           select_lambda::Symbol = :pijcv,
                           kwargs...)

    _validate_vcov_type(vcov_type)
    
    # Resolve penalty
    resolved_penalty = _resolve_penalty(penalty, model)
    penalty_config = build_penalty_config(model, resolved_penalty; lambda_init=lambda_init)
    use_penalty = has_penalties(penalty_config)
    
    # Build TPM bookkeeping
    books = build_tpm_mapping(model.data)
    
    # DISPATCH: Penalized vs Unpenalized
    if use_penalty && !isnothing(resolved_penalty)
        selector = _resolve_selector(select_lambda, penalty_config)
        
        if verbose
            println("Using penalized likelihood with $(penalty_config.n_lambda) smoothing parameter(s)")
        end
        
        return _fit_markov_panel_penalized(
            model, books, penalty_config, selector;
            constraints=constraints, verbose=verbose, solver=solver,
            vcov_type=vcov_type, vcov_threshold=vcov_threshold,
            loo_method=loo_method, lambda_init=lambda_init, kwargs...
        )
    end
    
    # UNPENALIZED PATH (existing code)
    # ... existing _fit_markov_panel implementation ...
end
```

---

## 6. MCEM Specifications

### 6.1 Overview

MCEM presents unique challenges for λ selection:
1. **Monte Carlo variance**: Likelihood estimates have MC error
2. **Iterative algorithm**: λ must be selected WITHIN iterations, not after convergence
3. **Computational cost**: Each iteration requires λ selection + M-step

**CORRECT Strategy**: Select λ within each MCEM iteration using current paths/weights.

**WHY**: Paths sampled at β_current are valid for evaluating Q(β) at ANY β via importance
weighting. This means within each iteration, we can optimize over (λ, β) jointly. We cannot
defer λ selection to after convergence because paths from β̂(λ_init) are not valid for
evaluating β̂(λ_trial) at different λ.

### 6.2 Updated `_fit_mcem` Signature

Add `select_lambda` kwarg:

```julia
function _fit_mcem(model::MultistateModel; 
                   # ... existing kwargs ...
                   penalty = :auto, 
                   lambda_init = 1.0,
                   select_lambda::Symbol = :none,  # NEW: :none, :pijcv, :efs, etc.
                   kwargs...)
```

### 6.3 MCEM with λ Selection (CORRECT: Within Each Iteration)

**Pseudocode for updated `_fit_mcem`**:

```julia
function _fit_mcem(model; penalty=:auto, lambda_init=1.0, select_lambda=:none, ...)
    # Existing initialization...
    resolved_penalty = _resolve_penalty(penalty, model)
    penalty_config = build_penalty_config(model, resolved_penalty; lambda_init=lambda_init)
    use_penalty = has_penalties(penalty_config)
    
    # Resolve selector
    selector = _resolve_selector(select_lambda, penalty_config)
    use_selection = !(selector isa NoSelection) && use_penalty
    
    β_current = get_parameters_flat(model)
    λ_current = get_hyperparameters(penalty_config)
    
    while keep_going
        # =================================================================
        # E-step: Sample paths given current β
        # =================================================================
        paths, weights = sample_paths(model, β_current, surrogate)
        # paths[i][j] = j-th sampled path for subject i
        # weights[i][j] = normalized importance weight
        
        # =================================================================
        # λ-step: Select λ given current paths/weights (NEW)
        # =================================================================
        if use_selection
            # Compute importance-weighted subject gradients/Hessians at β_current
            subject_grads, subject_hessians = compute_subject_grads_hessians_mcem(
                β_current, model, paths, weights
            )
            
            # Select λ to minimize PIJCV criterion
            # Inner loop: for each trial λ, fit β̂(λ) using SAME paths/weights
            λ_result = _select_lambda_mcem(
                model, paths, weights, penalty_config, selector;
                beta_init=β_current, subject_grads=subject_grads, 
                subject_hessians=subject_hessians
            )
            
            λ_current = λ_result.lambda
            penalty_config = set_hyperparameters(penalty_config, λ_current)
            β_warmstart = λ_result.warmstart_beta
        else
            β_warmstart = β_current
        end
        
        # =================================================================
        # M-step: Optimize β given current λ and paths/weights
        # =================================================================
        sm_data = SMPanelData(model, paths, weights)
        
        β_new = argmin_β [ -Q(β; paths, weights) + penalty(β, λ_current) ]
        # where Q(β) = Σ_i Σ_j w_ij * log L(β; path_ij)
        
        # =================================================================
        # Check convergence
        # =================================================================
        if converged(β_new, β_current, λ_current)
            β_current = β_new
            break
        end
        
        β_current = β_new
        update_surrogate!(surrogate, β_current)
    end
    
    # Compute final EDF
    edf = compute_edf_mcem(β_current, λ_current, penalty_config, paths, weights)
    
    return MultistateModelFitted(
        ...,
        smoothing_parameters = λ_current,
        edf = edf
    )
end
```

### 6.4 λ Selection Within MCEM Iteration

**Key Insight**: Within each iteration, paths/weights are FIXED. This makes λ selection
well-defined because:
- Q(β; paths, weights) is a valid estimate of E[log L(β | Z)] for ANY β
- Finding optimal λ for this fixed Q is the same nested optimization as exact/Markov data
- The only difference is that gradients/Hessians are importance-weighted

```julia
function _select_lambda_mcem(model, paths, weights, penalty, selector;
                              beta_init, subject_grads, subject_hessians)
    
    # Define PIJCV criterion using current paths/weights
    function pijcv_criterion(log_λ)
        λ = exp.(log_λ)
        
        # Inner: fit β̂(λ) using penalized M-step with SAME paths/weights
        penalty_trial = set_hyperparameters(penalty, λ)
        sm_data = SMPanelData(model, paths, weights)
        
        β_at_λ = argmin_β [ -Q(β; paths, weights) + penalty(β, λ) ]
        
        # Recompute importance-weighted grads/Hessians at β̂(λ) using SAME paths/weights
        g_i(λ), H_i(λ) = compute_subject_grads_hessians_mcem(β_at_λ, model, paths, weights)
        
        # Build penalized Hessian: H_λ = Σ_i H_i + λS
        H_λ = sum(H_i(λ)) + penalty_hessian(λ)
        
        # PIJCV: V = Σ_i D_i(β̂^{-i})
        V = 0
        for i in 1:n_subjects
            # LOO Newton step: β̂^{-i} ≈ β̂ + (H_λ - H_i)^{-1} g_i
            Δ_i = solve(H_λ - H_i(λ)[i], g_i(λ)[:, i])
            β_loo = β_at_λ + Δ_i
            
            # Evaluate subject i's expected loss at LOO params using SAME paths/weights
            D_i = -Σ_j weights[i][j] * log L(β_loo; paths[i][j])
            V += D_i
        end
        return V
    end
    
    # Optimize criterion over log(λ)
    log_λ_optimal = minimize(pijcv_criterion, log(λ_init); bounds=[-8, 8])
    λ_optimal = exp.(log_λ_optimal)
    
    # Get β̂ at optimal λ for warm-starting M-step
    β_warmstart = argmin_β [ -Q(β; paths, weights) + penalty(β, λ_optimal) ]
    
    return HyperparameterSelectionResult(
        lambda = λ_optimal,
        warmstart_beta = β_warmstart,
        ...
    )
end
```

### 6.5 New Data Type: `MCEMSelectionData`

For λ selection within MCEM, we need a data container that represents the importance-weighted complete-data likelihood:

```julia
"""
    MCEMSelectionData

Data container for λ selection within MCEM.

This wraps the sampled paths and importance weights from the E-step
for use in hyperparameter selection functions.
"""
struct MCEMSelectionData
    model::MultistateModel
    paths::Vector{Vector{SamplePath}}
    weights::Vector{Vector{Float64}}
end
```

### 6.6 New Function: `_select_hyperparameters_mcem`

**File**: `src/inference/smoothing_selection.jl`

```julia
function _select_hyperparameters(
    model::MultistateProcess,
    data::MCEMSelectionData,    # <-- Dispatch on MCEMSelectionData
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
) -> HyperparameterSelectionResult
```

**Implementation Notes**:
- Uses importance-weighted likelihood from MCEM
- Inner coefficient fitting uses `loglik_semi_markov_penalized`
- MC variance may affect selection stability (documented limitation)

### 6.6 New Function: `_fit_inner_coefficients_mcem`

```julia
function _fit_inner_coefficients(
    model::MultistateProcess,
    data::MCEMSelectionData,    # <-- Dispatch on MCEMSelectionData
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int = 50
) -> Vector{Float64}
    
    # Create SMPanelData for likelihood computation
    sm_data = SMPanelData(model, data.paths, data.weights)
    
    function penalized_nll(β, p)
        nll = loglik_semi_markov(β, sm_data; neg=true)
        pen = compute_penalty(β, penalty)
        return nll + pen
    end
    
    # Optimize with Ipopt
    adtype = SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=lb, ub=ub)
    
    sol = solve(prob, IpoptOptimizer(...); maxiters=maxiter, ...)
    return sol.u
end
```

### 6.7 MCEM Variance Warning

Similar to exact data, add warning for model-based variance with penalties:

```julia
# In _fit_mcem, before variance computation:
if vcov_type == :model && use_penalty
    @warn """Model-based variance (`vcov_type=:model`) may be inappropriate for penalized MCEM models.
    Consider using `vcov_type=:ij` (infinitesimal jackknife) for robust variance estimates."""
end
```

---

## 7. File Changes

### 7.1 Summary Table

| File | Changes | Lines Est. |
|------|---------|-----------|
| `src/inference/fit_markov.jl` | Add dispatcher pattern, penalty kwargs | +40 |
| `src/inference/fit_markov_penalized.jl` (NEW) | `_fit_markov_panel_penalized` | +200 |
| `src/inference/fit_mcem.jl` | Add `select_lambda`, post-convergence selection | +100 |
| `src/likelihood/loglik_markov.jl` | Add `loglik_markov_penalized` | +30 |
| `src/inference/smoothing_selection.jl` | Add data-type dispatch overloads | +200 |
| `src/types/data_containers.jl` | Add `MCEMSelectionData` | +30 |
| Tests | Unit and integration tests | +300 |
| **TOTAL** | | ~900 |

### 7.2 New Files

1. `src/inference/fit_markov_penalized.jl` - Penalized Markov panel fitting

### 7.3 Modified Files

1. `src/inference/fit_markov.jl` - Add dispatcher pattern
2. `src/inference/fit_mcem.jl` - Add λ selection
3. `src/likelihood/loglik_markov.jl` - Add penalized likelihood
4. `src/inference/smoothing_selection.jl` - Add data-type overloads
5. `src/types/data_containers.jl` - Add MCEMSelectionData
6. `src/MultistateModels.jl` - Add new file includes

---

## 8. Implementation Phases

### Phase M1: Markov Panel Foundation (~2-3 hours) ✅ COMPLETE

**Goal**: Basic penalized fitting for Markov panel data with fixed λ.

**Actions**:
- [x] **M1.1** Add `loglik_markov_penalized` to `loglik_markov.jl`
- [x] **M1.2** Add penalty kwargs to `_fit_markov_panel`
- [x] **M1.3** Add dispatch to penalized path when penalty active
- [x] **M1.4** Create `_fit_markov_panel_penalized` (NoSelection only)
- [x] **M1.5** Verify fixed-λ fitting works

**Completion Criteria**:
- `fit(markov_model; penalty=SplinePenalty(), select_lambda=:none)` works ✅
- Returns proper `smoothing_parameters` and `edf` ✅
- All existing Markov tests pass ✅

---

### Phase M2: Markov Panel λ Selection (~3-4 hours) ✅ COMPLETE

**Goal**: Add automatic λ selection for Markov panel.

**Actions**:
- [x] **M2.1** Add `_fit_inner_coefficients` overload for `MPanelData`
- [x] **M2.2** Add `_select_hyperparameters` overload for `MPanelData`
- [x] **M2.3** Add `compute_edf_markov` function
- [x] **M2.4** Add subject gradient/Hessian computation for Markov
- [x] **M2.5** Wire up λ selection in `_fit_markov_panel_penalized`
- [x] **M2.6** Add variance warning for penalized Markov

**Completion Criteria**:
- `fit(markov_model; penalty=SplinePenalty(), select_lambda=:pijcv)` works ✅
- λ is automatically selected ✅
- EDF is computed correctly ✅
- All tests pass ✅

---

### Phase M3: MCEM Fixed λ Verification (~1-2 hours) ✅ COMPLETE

**Goal**: Verify existing MCEM penalty support works with new infrastructure.

**Actions**:
- [x] **M3.1** Verify existing penalty handling uses new types
- [x] **M3.2** Ensure `smoothing_parameters` and `edf` are returned
- [x] **M3.3** Add variance warning for penalized MCEM
- [x] **M3.4** Add tests for MCEM with fixed λ (test_mcem_splines_basic.jl)

**Completion Criteria**:
- `fit(mcem_model; penalty=SplinePenalty(), lambda_init=10.0)` works ✅
- Returns correct `smoothing_parameters` ✅
- Warning emitted for `vcov_type=:model` ✅

---

### Phase M4: MCEM λ Selection (~4-6 hours) ✅ COMPLETE

**Goal**: Add within-iteration λ selection for MCEM.

**Actions**:
- [x] **M4.1** Add `MCEMSelectionData` type (data_containers.jl)
- [x] **M4.2** Add `_fit_inner_coefficients` overload for `MCEMSelectionData`
- [x] **M4.3** Add `_select_hyperparameters` overload for `MCEMSelectionData`
- [x] **M4.4** Add `compute_edf_mcem` function
- [x] **M4.5** Add within-iteration λ selection logic to `_fit_mcem` (select_lambda kwarg)
- [x] **M4.6** Bug fix: edf_trace extracts .total from NamedTuple (fit_mcem.jl:761)

**Completion Criteria**:
- `fit(mcem_model; penalty=SplinePenalty(), select_lambda=:pijcv)` works ✅
- λ selected within MCEM iterations using PIJCV ✅
- All tests pass ✅

---

### Phase M5: Cleanup and Documentation (~2 hours) ✅ COMPLETE

**Goal**: Finalize, document, and test.

**Actions**:
- [x] **M5.1** Update docstrings for all new functions (MCEMSelectionData, _nested_optimization_pijcv_mcem, compute_edf_mcem)
- [x] **M5.2** Add integration tests (test_mcem_splines_basic.jl, test_mcem_lambda_selection.jl)
- [x] **M5.3** Run full test suite - 2097 tests pass
- [x] **M5.4** Update this document with completion status

**Completion Criteria**:
- All tests pass ✅ (2097 tests)
- Docstrings complete ✅
- No dead code ✅

---

## 9. Testing Requirements

### 9.1 Unit Tests

| Test | Purpose | Status |
|------|---------|--------|
| `test_markov_penalized_likelihood.jl` | Test `loglik_markov_penalized` | ✅ |
| `test_markov_inner_coefficients.jl` | Test `_fit_inner_coefficients` for MPanelData | ✅ |
| `test_mcem_selection_data.jl` | Test `MCEMSelectionData` construction | ✅ |

### 9.2 Integration Tests

| Test | Purpose | Status |
|------|---------|--------|
| `test_fit_markov_penalized_fixed.jl` | Markov with fixed λ | ✅ |
| `test_fit_markov_penalized_pijcv.jl` | Markov with PIJCV selection | ✅ |
| `test_mcem_splines_basic.jl` | MCEM with unpenalized splines | ✅ |
| `test_mcem_lambda_selection.jl` | MCEM with PIJCV λ selection | ✅ |

### 9.3 Regression Tests

Run after each phase:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

All 2097+ tests must pass. ✅ **VERIFIED 2026-01-24**

---

## 10. Agent Handoff Protocol

### 10.1 Session Start Protocol

1. Read `.github/skills/codebase-knowledge/SKILL.md`
2. Read this document (`scratch/PENALIZED_MARKOV_MCEM_REFACTORING_PLAN.md`)
3. Read latest handoff document (if exists)
4. Verify current state matches handoff
5. Continue from next uncompleted action

### 10.2 Phase Boundaries

**Critical**: Do not start a new phase until previous is complete.

**Phase dependencies**:
- Phase M2 requires Phase M1 complete
- Phase M4 requires Phase M3 complete
- Phase M5 requires all previous phases complete

### 10.3 Handoff Document Template

Create `scratch/PENALIZED_MARKOV_MCEM_HANDOFF_<DATE>.md` with:
- Current phase and status
- Completed actions
- Files modified
- Tests passing (count)
- Next actions
- Any blockers

---

## Appendix A: Key Type Signatures

### A.1 Data Container Hierarchy

```julia
# Exact data (continuous observation)
ExactData                    # From exact refactoring

# Markov panel data
MPanelData                   # Existing, for loglik_markov

# Semi-Markov panel data (MCEM)
SMPanelData                  # Existing, for loglik_semi_markov

# NEW: MCEM selection data
MCEMSelectionData            # For λ selection within MCEM
```

### A.2 Dispatch Pattern for `_fit_inner_coefficients`

```julia
# Exact data (existing)
_fit_inner_coefficients(model, data::ExactData, penalty, beta; ...)

# Markov panel (new)
_fit_inner_coefficients(model, data::MPanelData, penalty, beta; ...)

# MCEM selection (new)
_fit_inner_coefficients(model, data::MCEMSelectionData, penalty, beta; ...)
```

### A.3 Dispatch Pattern for `_select_hyperparameters`

```julia
# Exact data (existing)
_select_hyperparameters(model, data::ExactData, penalty, selector; ...)

# Markov panel (new)
_select_hyperparameters(model, data::MPanelData, penalty, selector; ...)

# MCEM selection (new)
_select_hyperparameters(model, data::MCEMSelectionData, penalty, selector; ...)
```

---

## Appendix B: Validation Criteria

### B.1 Markov Panel Validation

1. **Parameter recovery**: Fit simulated Markov data with known spline, verify recovery
2. **λ selection**: Compare PIJCV-selected λ to grid search minimum
3. **EDF**: Verify EDF ≈ trace(F) where F is the smoother matrix

### B.2 MCEM Validation

1. **Fixed λ**: Verify MCEM converges to same estimate as exact data (large ESS)
2. **λ selection**: Verify post-convergence λ is reasonable
3. **MC stability**: Run multiple times, check λ selection stability

---

**END OF PLAN**
