# Workstream C: MCEM Adaptation for Performance Iteration

**Date**: 2026-01-30  
**Status**: Analysis Complete, Implementation Plan Ready  
**Author**: GitHub Copilot (Claude Opus 4.5)

---

## Executive Summary

This document analyzes the current MCEM smoothing parameter selection infrastructure and proposes specific modifications to handle Monte Carlo noise. The key findings are:

1. **λ Trace**: Already implemented via `lambda_trace` and `edf_trace` in `fit_mcem.jl`
2. **λ Dampening**: Requires new MCEM-specific dampening with exponential moving average
3. **EDF for MCEM (Issue 8.0.2)**: Current implementation uses standard EDF formula; Louis's identity-based EDF would provide proper MC variance accounting

---

## Part 1: Current State Analysis

### 1.1 File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `src/inference/fit_mcem.jl` | Main MCEM fitting loop | 1461 |
| `src/inference/smoothing_selection/dispatch_mcem.jl` | MCEM hyperparameter selection | 858 |
| `src/inference/smoothing_selection/implicit_diff.jl` | Implicit differentiation (includes MCEM) | 2500+ |
| `src/inference/smoothing_selection/common.jl` | Shared utilities including `adaptive_dampening` | 1200 |

### 1.2 λ Selection Flow in MCEM

```
fit_mcem.jl (main loop)
  └── _select_hyperparameters(model, MCEMSelectionData, penalty, selector)
        ├── PIJCVSelector → _nested_optimization_pijcv_mcem() 
        │                 → _nested_optimization_pijcv_mcem_implicit()
        ├── REMLSelector  → _nested_optimization_criterion_mcem(:efs)
        └── PERFSelector  → _nested_optimization_criterion_mcem(:perf)
```

### 1.3 Existing λ Trace Implementation

In `fit_mcem.jl` (lines 475-477):
```julia
# Track λ history for convergence records
lambda_trace = use_lambda_selection ? ElasticArray{Float64, 2}(undef, penalty_config.n_lambda, 0) : nothing
edf_trace = use_lambda_selection ? Vector{Float64}() : nothing
```

In MCEM main loop (lines 803-807):
```julia
# Track λ and EDF (extract total from NamedTuple)
append!(lambda_trace, lambda_result.lambda)
push!(edf_trace, lambda_result.edf.total)
```

In convergence records (lines 1215-1220):
```julia
# Add λ selection records to convergence records if used
if return_convergence_records && use_lambda_selection && !isnothing(lambda_trace)
    lambda_selection_records = (
        lambda_trace = lambda_trace,
        edf_trace = edf_trace
    )
    ConvergenceRecords = merge(ConvergenceRecords, lambda_selection_records)
end
```

**Finding**: λ trace is already implemented. No changes needed for basic tracing.

### 1.4 Current λ Warmstarting

In `fit_mcem.jl` (lines 479-481):
```julia
# λ warmstart: Pass previous iteration's λ as starting point for next iteration
# This prevents oscillating λ values when importance weights change between iterations
lambda_warmstart = nothing  # Will be set after first λ selection
```

After λ selection (lines 801-802):
```julia
# Store λ for warmstarting next iteration
lambda_warmstart = lambda_result.lambda
```

**Finding**: Basic warmstarting exists but lacks exponential moving average dampening.

---

## Part 2: Proposed Changes

### 2.1 λ Moving Average/Dampening (Priority: HIGH)

**Problem**: MC noise in gradients/Hessians causes λ to oscillate between MCEM iterations.

**Solution**: Implement exponential moving average for λ updates:

```julia
λ_smoothed = γ × λ_new + (1 - γ) × λ_old
```

where γ decreases as MCEM converges (controlled by iteration or ESS).

**Implementation Location**: `fit_mcem.jl` after λ selection result

**Proposed Code**:

```julia
# In fit_mcem.jl, after lambda_result = _select_hyperparameters(...)

# --- BEGIN NEW CODE ---
# MCEM-specific λ dampening via exponential moving average
# γ decreases as MCEM progresses to stabilize λ near convergence
if iter == 1
    # First iteration: use selected λ directly
    lambda_smoothed = lambda_result.lambda
else
    # Compute adaptive smoothing factor:
    # - Early iterations (high MC variance): small γ (more smoothing)
    # - Later iterations (lower MC variance): larger γ (less smoothing)
    # Formula: γ = max(0.3, min(0.9, 0.3 + 0.05 * iter))
    gamma = clamp(0.3 + 0.05 * iter, 0.3, 0.9)
    
    # Optional: Reduce γ if oscillation detected in λ trace
    if use_lambda_selection && length(lambda_trace) >= 4
        # Check for sign changes in log(λ) differences
        recent_log_lambda = [log.(lambda_trace[:, i]) for i in max(1, size(lambda_trace, 2)-3):size(lambda_trace, 2)]
        if _detect_lambda_oscillation(recent_log_lambda)
            gamma *= 0.5  # Increase smoothing
            verbose && @info "λ oscillation detected, reducing γ to $(round(gamma, digits=2))"
        end
    end
    
    # Apply exponential moving average
    lambda_smoothed = gamma .* lambda_result.lambda .+ (1 - gamma) .* lambda_warmstart
end

# Use smoothed λ for penalty config update
penalty_config = set_hyperparameters(penalty_config, lambda_smoothed)
lambda_warmstart = lambda_smoothed  # Update warmstart for next iteration
# --- END NEW CODE ---
```

**Helper Function** (add to `common.jl`):

```julia
"""
    _detect_lambda_oscillation(log_lambda_history::Vector{Vector{Float64}}; threshold::Int=2) -> Bool

Detect oscillation in λ trace by counting sign changes in log-scale differences.
"""
function _detect_lambda_oscillation(log_lambda_history::Vector{Vector{Float64}}; threshold::Int=2)
    length(log_lambda_history) < 3 && return false
    
    # Compute differences for each λ component
    n_lambda = length(first(log_lambda_history))
    total_sign_changes = 0
    
    for j in 1:n_lambda
        diffs = [log_lambda_history[i+1][j] - log_lambda_history[i][j] for i in 1:length(log_lambda_history)-1]
        for i in 1:length(diffs)-1
            if diffs[i] * diffs[i+1] < 0
                total_sign_changes += 1
            end
        end
    end
    
    return total_sign_changes >= threshold
end
```

### 2.2 Enhanced λ Trace (Priority: LOW - Already Implemented)

**Current State**: `lambda_trace` and `edf_trace` are already tracked and included in convergence records.

**Potential Enhancement**: Add λ variance tracking for diagnostics.

```julia
# Add to convergence_records initialization
lambda_variance_trace = use_lambda_selection ? Vector{Float64}() : nothing

# In MCEM loop, compute running variance of λ
if iter >= 3 && use_lambda_selection
    recent_lambdas = [lambda_trace[:, i] for i in max(1, size(lambda_trace, 2)-4):size(lambda_trace, 2)]
    lambda_var = mean(var.(recent_lambdas))
    push!(lambda_variance_trace, lambda_var)
end
```

### 2.3 Proper EDF for MCEM (Issue 8.0.2) (Priority: MEDIUM)

**Problem**: Current `compute_edf_mcem()` uses the standard formula:
```
EDF = tr(H_unpenalized × H_λ⁻¹)
```

This doesn't account for Monte Carlo variance in the Hessian estimate.

**Louis's Identity Approach**:
For MCEM, the observed Fisher information is:
```
I_obs = E[I_complete | Y_obs] - Var[S_complete | Y_obs]
```

The current code estimates `E[I_complete | Y_obs]` but doesn't subtract the missing information term.

**Analysis of Current Implementation** (`dispatch_mcem.jl`, lines 748-797):

```julia
function compute_edf_mcem(beta::Vector{Float64}, lambda::Vector{Float64}, 
                          penalty_config::AbstractPenalty, data::MCEMSelectionData)
    # Compute importance-weighted Hessian
    subject_hessians_ll = compute_subject_hessians(beta, data.model, data.paths, data.weights)
    subject_hessians = [-H for H in subject_hessians_ll]
    H_unpenalized = sum(subject_hessians)
    
    # ... standard EDF computation ...
```

**Problem**: `H_unpenalized` is computed as:
```
H_unpenalized = Σᵢ Σⱼ wᵢⱼ (-∇²log f(Zᵢⱼ; β))
```

This is the importance-weighted average of complete-data Hessians, which estimates `E[I_complete | Y]` but doesn't subtract the missing information.

**Proposed Solution**: Implement `compute_edf_mcem_louis()`:

```julia
"""
    compute_edf_mcem_louis(beta, lambda, penalty_config, data::MCEMSelectionData) -> NamedTuple

Compute effective degrees of freedom for MCEM using Louis's identity.

The observed Fisher information is:
    I_obs = E[I_complete | Y] - Var[S_complete | Y]

where:
- E[I_complete | Y] = Σᵢ Σⱼ wᵢⱼ (-∇²log f(Zᵢⱼ; β))  [complete-data expected information]
- Var[S_complete | Y] = Σᵢ [Σⱼ wᵢⱼ Sᵢⱼ Sᵢⱼᵀ - (Σⱼ wᵢⱼ Sᵢⱼ)(Σⱼ wᵢⱼ Sᵢⱼ)ᵀ]  [missing information]

This gives a more accurate EDF estimate that accounts for the uncertainty
in the latent paths.

# Arguments
- `beta`: Current parameter estimate
- `lambda`: Smoothing parameters
- `penalty_config`: Penalty configuration
- `data`: MCEMSelectionData with paths and importance weights

# Returns
NamedTuple with:
- `total::Float64`: Total model EDF
- `per_term::Vector{Float64}`: Per-term EDF values
- `missing_info_trace::Float64`: Trace of missing information matrix (diagnostic)
"""
function compute_edf_mcem_louis(beta::Vector{Float64}, lambda::Vector{Float64}, 
                                penalty_config::AbstractPenalty, data::MCEMSelectionData)
    n_subjects = length(data.paths)
    n_params = length(beta)
    
    # Initialize accumulators
    I_complete = zeros(n_params, n_params)  # E[I_complete | Y]
    I_missing = zeros(n_params, n_params)   # Var[S_complete | Y]
    
    for i in 1:n_subjects
        weights = data.weights[i]
        npaths = length(data.paths[i])
        
        # Collect scores and Hessians for this subject
        scores = Matrix{Float64}(undef, n_params, npaths)
        
        for j in 1:npaths
            path = data.paths[i][j]
            
            # Compute score (gradient of log-likelihood)
            score_j = _compute_path_score(beta, path, data.model)
            scores[:, j] = score_j
            
            # Compute Hessian contribution
            hess_j = _compute_path_hessian(beta, path, data.model)
            I_complete .+= weights[j] * (-hess_j)
        end
        
        # Missing information for subject i (weighted outer products minus product of weighted means)
        weighted_scores = scores * Diagonal(weights)
        weighted_mean_score = weighted_scores * ones(npaths)  # Σⱼ wᵢⱼ Sᵢⱼ
        
        # Var[S | Y] = E[SSᵀ | Y] - E[S | Y]E[S | Y]ᵀ
        outer_prod_sum = weighted_scores * scores'  # Σⱼ wᵢⱼ Sᵢⱼ Sᵢⱼᵀ
        mean_outer_prod = weighted_mean_score * weighted_mean_score'
        
        I_missing .+= outer_prod_sum - mean_outer_prod
    end
    
    # Observed Fisher = Complete - Missing
    I_obs = I_complete - I_missing
    
    # Build penalized Hessian using observed Fisher
    H_lambda = _build_penalized_hessian(I_obs, lambda, penalty_config; beta=beta)
    
    # EDF = tr(I_obs × H_λ⁻¹)
    H_inv = try
        inv(Symmetric(H_lambda))
    catch
        n_terms = _count_penalty_terms(penalty_config)
        return (total = NaN, per_term = fill(NaN, n_terms), missing_info_trace = NaN)
    end
    
    A = I_obs * H_inv
    total_edf = tr(A)
    
    # Per-term EDF
    edf_vec = _compute_per_term_edf(A, penalty_config)
    
    return (total = total_edf, per_term = edf_vec, missing_info_trace = tr(I_missing))
end
```

**Note**: This requires helper functions `_compute_path_score` and `_compute_path_hessian` which would use ForwardDiff similar to existing code in `fit_mcem.jl`.

---

## Part 3: Implementation Roadmap

### Phase 1: λ Dampening (Recommended First)

1. Add `_detect_lambda_oscillation()` to `common.jl`
2. Modify `fit_mcem.jl` to apply exponential moving average after λ selection
3. Add `lambda_dampening_gamma` parameter to `MCEMConfig`
4. Test with existing MCEM spline tests

**Estimated effort**: 2-4 hours
**Files modified**: 
- `src/inference/fit_mcem.jl`
- `src/inference/smoothing_selection/common.jl`

### Phase 2: Enhanced Diagnostics (Optional)

1. Add `lambda_variance_trace` to convergence records
2. Add dampening history (γ values) to convergence records

**Estimated effort**: 1 hour
**Files modified**: `src/inference/fit_mcem.jl`

### Phase 3: Louis's Identity EDF (Research Required)

1. Implement `compute_edf_mcem_louis()`
2. Add `_compute_path_score()` and `_compute_path_hessian()` helpers
3. Benchmark against standard EDF to validate
4. Consider making this the default for MCEM

**Estimated effort**: 4-8 hours
**Files modified**:
- `src/inference/smoothing_selection/dispatch_mcem.jl`
- New file or addition to `common.jl`

---

## Part 4: Key Design Decisions

### Decision 1: Where to Apply Dampening

**Options**:
A. Inside `_select_hyperparameters()` (before returning)
B. In `fit_mcem.jl` after receiving result

**Recommendation**: Option B (in `fit_mcem.jl`)

**Rationale**: 
- Keeps selection functions pure (return optimal λ for given data)
- Dampening is specific to MCEM iteration context (not general λ selection)
- Allows different dampening strategies without modifying selection code

### Decision 2: Dampening Formula

**Options**:
A. Fixed γ (e.g., 0.5)
B. Iteration-dependent: γ = f(iter)
C. ESS-dependent: γ = f(ESS)
D. Adaptive based on oscillation detection

**Recommendation**: Combination of B and D
```julia
gamma_base = clamp(0.3 + 0.05 * iter, 0.3, 0.9)
gamma = oscillation_detected ? gamma_base * 0.5 : gamma_base
```

### Decision 3: Louis's Identity Implementation

**Options**:
A. Replace `compute_edf_mcem()` entirely
B. Add new function `compute_edf_mcem_louis()`, keep old as default
C. Add parameter to `compute_edf_mcem()` to select method

**Recommendation**: Option B initially, then consider C

**Rationale**: 
- Non-breaking change
- Allows comparison of methods
- Can validate Louis EDF against standard before switching default

---

## Part 5: Testing Strategy

### Unit Tests

```julia
# test_mcem_lambda_dampening.jl

@testset "MCEM λ Dampening" begin
    @testset "Oscillation Detection" begin
        # Test oscillation detection helper
        history = [[1.0], [2.0], [1.5], [2.5], [1.3]]  # Oscillating
        @test _detect_lambda_oscillation(history) == true
        
        history = [[1.0], [1.5], [2.0], [2.5], [3.0]]  # Monotone
        @test _detect_lambda_oscillation(history) == false
    end
    
    @testset "EMA Application" begin
        λ_old = [1.0, 2.0]
        λ_new = [2.0, 4.0]
        γ = 0.5
        λ_smoothed = γ .* λ_new .+ (1 - γ) .* λ_old
        @test λ_smoothed ≈ [1.5, 3.0]
    end
end
```

### Integration Tests

Run existing MCEM spline tests to ensure no regression:
```bash
julia --project=MultistateModelsTests -e 'include("MultistateModelsTests/unit/test_splines_mcem.jl")'
```

---

## Part 6: Summary of Changes

| Change | Priority | Files | Status |
|--------|----------|-------|--------|
| λ trace in convergence records | N/A | `fit_mcem.jl` | ✅ Already implemented |
| λ warmstarting | N/A | `fit_mcem.jl` | ✅ Already implemented |
| λ EMA dampening | HIGH | `fit_mcem.jl`, `common.jl` | 📋 Ready to implement |
| Oscillation detection | HIGH | `common.jl` | 📋 Ready to implement |
| λ variance trace | LOW | `fit_mcem.jl` | 📋 Optional |
| Louis's identity EDF | MEDIUM | `dispatch_mcem.jl` | 📋 Research required |

---

## Appendix: Relevant Code Locations

### Current λ Selection Entry Point
- `fit_mcem.jl:787-800` - calls `_select_hyperparameters()`

### Current EDF Computation
- `dispatch_mcem.jl:748-797` - `compute_edf_mcem()`

### Existing Dampening Infrastructure
- `common.jl:239-273` - `adaptive_dampening()` (for performance iteration)
- `common.jl:215-237` - `detect_oscillation()` (for V criterion history)

### Louis's Identity Reference Implementation
- `fit_mcem.jl:1343-1430` - Model-based variance uses Louis's identity for vcov
