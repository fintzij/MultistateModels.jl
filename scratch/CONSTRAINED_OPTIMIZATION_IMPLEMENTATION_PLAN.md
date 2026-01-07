# Constrained Optimization Implementation Plan

**Date**: 2025-01-06  
**Session**: 5 (continuation)  
**Goal**: Fix PIJCV λ selection by implementing box-constrained optimization with natural-scale parameters

---

## Executive Summary

The PIJCV criterion fails to select reasonable smoothing parameters because the exp-transform used to ensure non-negativity makes the penalty non-quadratic. The Newton LOO approximation `β̂⁻ⁱ ≈ β̂ + H_{λ,-i}⁻¹gᵢ` requires a quadratic objective. 

**Solution**: Store parameters on natural scale with box constraints (lb ≥ 0 for non-negative parameters), making the penalty truly quadratic: P(β) = (λ/2)βᵀSβ.

**Key Simplification**: This is an opportunity to **remove** the dual estimation/natural scale architecture entirely. After this change:
- Parameters are **always** stored on natural scale
- No log-transforms for storage (remove `exp_transform` logic)
- No `_transform_to_estimation` / `_transform_to_natural` functions needed
- Box constraints via Ipopt handle non-negativity directly

---

## Part 1: Current Architecture Analysis (TO BE REMOVED)

### 1.1 Parameter Flow Trace

```
User provides: natural-scale starting values
    ↓
multistatemodel(): Converts to estimation scale (log for non-negative params)
    ↓
model.parameters.flat: [log(θ₁), log(θ₂), ...] for Exp/Wei/Spline
                       [θ_shape, log(θ_rate)] for Gompertz
    ↓
optimizer: Works on flat vector (currently unconstrained)
    ↓
unflatten_natural(): Applies exp() to convert back to natural scale
    ↓
hazard_fn(): Receives natural-scale params, no internal transforms
```

### 1.2 Key Files and Their Roles (Changes Needed)

| File | Current Role | Change |
|------|--------------|--------|
| `src/utilities/transforms.jl` | Scale conversion | **REMOVE** or simplify to identity |
| `src/hazard/generators.jl` | Hazard code gen | No change (already expect natural scale) |
| `src/construction/multistatemodel.jl` | Model build, splines | **REMOVE** exp() from `_spline_ests2coefs` |
| `src/types/infrastructure.jl` | Penalty computation | **REMOVE** `exp_transform` field and logic |
| `src/inference/fit_exact.jl` | Fitting entry | **ADD** lb/ub to OptimizationProblem |
| `src/inference/smoothing_selection.jl` | λ selection | **ADD** lb/ub, penalty becomes truly quadratic |

### 1.3 Current exp-Transform Locations (TO BE REMOVED)

1. **Parametric hazards** (`transforms.jl` lines 40-56):
   - Exponential: `exp(log_rate)` → rate  **REMOVE**
   - Weibull: `exp(log_shape)`, `exp(log_scale)` → shape, scale  **REMOVE**
   - Gompertz: `shape` (identity), `exp(log_rate)` → shape, rate  **REMOVE exp for rate**

2. **Spline hazards** (`multistatemodel.jl` line 665):
   - `_spline_ests2coefs`: `exp.(ests)` for monotone==0  **REMOVE**

3. **Penalty computation** (`infrastructure.jl` line 550):
   - `compute_penalty`: `β_j = term.exp_transform ? exp.(θ_j) : θ_j`  **REMOVE conditional, always use θ_j directly**

### 1.4 The Core Problem

The penalty function `compute_penalty` does:
```julia
β_j = term.exp_transform ? exp.(θ_j) : θ_j
penalty += λ * dot(β_j, S * β_j)
```

This means P(θ) = (λ/2) exp(θ)ᵀ S exp(θ), which is **NOT** quadratic in θ.

The PIJCV Newton approximation assumes:
```
β̂⁻ⁱ ≈ β̂ + H_{λ,-i}⁻¹ gᵢ  (valid only for quadratic objectives)
```

When the penalty is non-quadratic, this approximation breaks down, causing PIJCV to select extreme λ values.

**After this change**: `compute_penalty` will simply be `penalty += λ * dot(θ_j, S * θ_j)` — truly quadratic.

---

## Part 2: Proposed Architecture (SIMPLIFIED)

### 2.1 Core Change

**Store parameters on natural scale, use box constraints for non-negativity**

```
Before: θ ∈ ℝ (unconstrained), non-negativity via exp(θ)
After:  β ∈ [0, ∞) (box constrained), β stored directly
```

### 2.2 Bounds Generation by Hazard Family

| Family | Parameter | Natural Scale | Lower Bound | Upper Bound |
|--------|-----------|---------------|-------------|-------------|
| Exponential | rate | β ≥ 0 | 0 | Inf |
| Weibull | shape | β ≥ 0 | 0 | Inf |
| Weibull | scale | β ≥ 0 | 0 | Inf |
| Gompertz | shape | β ∈ ℝ | -Inf | Inf |
| Gompertz | rate | β ≥ 0 | 0 | Inf |
| Spline | coefs | β ≥ 0 | 0 | Inf |
| Covariate | any | β ∈ ℝ | -Inf | Inf |

**Note**: Lower bound is 0 (non-negativity constraint, not strict positivity).

### 2.3 Design Questions

**Q1: Where to store bounds?**

Option A: In `model.parameters` NamedTuple
```julia
parameters = (
    flat = [...],
    lb = [...],  # NEW
    ub = [...],  # NEW
    nested = ...,
    natural = ...,
    reconstructor = ...
)
```

Option B: Compute bounds on-demand from hazard metadata
```julia
function generate_parameter_bounds(model::MultistateProcess) -> (lb, ub)
```

**Recommendation**: Option B is cleaner - bounds are deterministic given hazard structure, no need to store them. Generate bounds in fitting functions.

**Q2: How to handle monotone splines?**

For monotone splines (`monotone ∈ {-1, 1}`), the I-spline transformation is:
```julia
coefs[i] = coefs[i-1] + exp(θ[i]) * (t[i+k] - t[i]) / k
```

The increments `exp(θ[i])` must be non-negative. With constrained optimization:
- Store increments directly on natural scale (non-negative)
- Use lb ≥ 0 for increment parameters

**Action**: Simplify `_spline_ests2coefs` to always use identity (no exp).

**Q3: Backward compatibility?**

**No backward compatibility mode.** This is a breaking change that simplifies the codebase. All parameters will be on natural scale with box constraints. The old log-transform approach is removed entirely.

### 2.4 User vs Package Constraint Combination (CRITICAL)

**Problem**: Users may provide their own box constraints on parameters that already have package-level constraints (e.g., spline coefficients must be non-negative). How do we combine them safely?

**Existing Constraint System**: The current `constraints` argument to `fit()` handles **nonlinear functional constraints** (e.g., `h12_rate < 2 * h13_rate`) via `make_constraints()`. These use `lcons`/`ucons` for the constraint function outputs, NOT parameter bounds.

**New Box Constraint System**: We're adding `lb`/`ub` for **simple box constraints** on parameters directly.

**Constraint Classification**:

| Type | Source | Example | Relaxable? |
|------|--------|---------|------------|
| **Hard** (package) | Hazard family | Spline coefs ≥ 0 | NO - breaks model |
| **Soft** (user) | User preference | Rate ∈ [0.01, 100] | YES - user choice |

**Combination Rule: Intersection (Most Restrictive)**

```julia
final_lb = max.(package_lb, user_lb)
final_ub = min.(package_ub, user_ub)
```

This ensures:
1. User CANNOT loosen safety constraints (e.g., make spline coefs unbounded)
2. User CAN tighten bounds (e.g., restrict rate to [0.01, 100])
3. User CAN add bounds to unconstrained parameters (e.g., covariate coefficients)

**Validation at Combination Time**:
```julia
# Check bounds are sensible
if any(final_lb .> final_ub)
    error("User bounds conflict with package constraints: lb > ub for parameters ...")
end

# Check initial values satisfy combined bounds  
if any(init .< final_lb) || any(init .> final_ub)
    error("Initial values violate combined bounds")
end
```

**API Design**:

```julia
# User provides box constraints via Dict with parameter names (required for safety)
fit(model;
    box_bounds = Dict(
        :h12_rate => (lb=0.01, ub=100.0),
        :h12_x => (lb=-5.0, ub=5.0)
    ),
    ...
)

# Can specify lb only, ub only, or both
box_bounds = Dict(
    :h12_rate => (lb=0.01,),           # Only lower bound
    :h12_x => (ub=5.0,),               # Only upper bound  
    :h13_rate => (lb=0.01, ub=100.0)   # Both bounds
)
```

**Note**: Vector-based bounds API intentionally NOT supported. With many parameters, positional
specification is error-prone. Named parameters via Dict are safer and self-documenting.
```

```julia
"""
    generate_parameter_bounds(model::MultistateProcess; 
                              user_bounds=nothing) -> (lb, ub)

Generate combined lower and upper bounds for box-constrained optimization.

Package bounds (hard constraints) are determined by hazard family:
- Non-negative parameters (rates, shapes, spline coefs): lb = 0
- Unconstrained parameters (Gompertz shape, covariates): lb = -Inf

User bounds are combined via intersection: final = max(pkg, user) for lb, min for ub.

# Arguments
- `model`: MultistateProcess with hazard definitions
- `user_bounds`: Optional Dict{Symbol, NamedTuple} mapping parameter names to bounds
  - Each entry: `:param_name => (lb=..., ub=...)` (lb and/or ub)
  - Example: `Dict(:h12_rate => (lb=0.01, ub=100.0), :h12_x => (ub=5.0,))`

# Returns
- `(lb::Vector{Float64}, ub::Vector{Float64})`: Combined bounds

# Throws
- `ArgumentError`: If user bounds conflict with package constraints (lb > ub)
- `ArgumentError`: If unknown parameter name provided
"""
function generate_parameter_bounds(model::MultistateProcess; 
                                   user_bounds=nothing)
    # 1. Generate package bounds from hazard families
    pkg_lb, pkg_ub = _generate_package_bounds(model)
    
    # 2. If no user bounds, return package bounds
    if isnothing(user_bounds)
        return pkg_lb, pkg_ub
    end
    
    # 3. Convert user bounds Dict to vectors
    user_lb_vec, user_ub_vec = _resolve_user_bounds(user_bounds, model)
    
    # 4. Combine via intersection
    final_lb = max.(pkg_lb, user_lb_vec)
    final_ub = min.(pkg_ub, user_ub_vec)
    
    # 5. Validate combined bounds
    conflicts = findall(final_lb .> final_ub)
    if !isempty(conflicts)
        parnames = _get_flat_parnames(model)
        conflict_names = parnames[conflicts]
        throw(ArgumentError(
            "User bounds conflict with package constraints for parameters: $(conflict_names). " *
            "Package requires lb=$(pkg_lb[conflicts]), ub=$(pkg_ub[conflicts]). " *
            "User specified lb=$(user_lb_vec[conflicts]), ub=$(user_ub_vec[conflicts])."
        ))
    end
    
    return final_lb, final_ub
end
```

**Edge Cases**:

1. **User tries to remove non-negativity**: `user_lb = -Inf` for spline coef
   - Result: `final_lb = max(0, -Inf) = 0` ✓ (package wins)

2. **User tightens non-negative bound**: `user_lb = 0.01` for rate (pkg: 0)
   - Result: `final_lb = max(0, 0.01) = 0.01` ✓ (user wins)

3. **User adds upper bound**: `user_ub = 100` for rate (pkg: Inf)
   - Result: `final_ub = min(Inf, 100) = 100` ✓ (user wins)

4. **Conflicting bounds**: `user_lb = 10, user_ub = 5`
   - Result: Error with clear message ✓

5. **User bounds on covariate**: `user_lb = -5, user_ub = 5` for β
   - Result: `final_lb = max(-Inf, -5) = -5`, `final_ub = min(Inf, 5) = 5` ✓

---

## Part 3: Implementation Steps (Dependency Order)

### Phase 1: Bounds Infrastructure

#### Step 1.1: Create bounds generation module
**File**: `src/utilities/bounds.jl` (NEW)

```julia
# =============================================================================
# Parameter Bounds for Box-Constrained Optimization
# =============================================================================

# Default lower bound for non-negative parameters
const NONNEG_LB = 0.0

"""
    _generate_package_bounds(model::MultistateProcess) -> (lb, ub)

Generate package-level (hard) bounds based on hazard family.
These are the minimum constraints required for model validity.
"""
function _generate_package_bounds(model::MultistateProcess)
    n_params = length(model.parameters.flat)
    lb = fill(-Inf, n_params)
    ub = fill(Inf, n_params)
    
    param_idx = 1
    for hazard in model.hazards
        family = hazard.family
        n_baseline = hazard.npar_baseline
        n_covar = hazard.npar_total - n_baseline
        
        # Set bounds for baseline parameters based on family
        baseline_lb = _get_baseline_lb(family, n_baseline)
        lb[param_idx:param_idx+n_baseline-1] .= baseline_lb
        
        # Covariate coefficients are unconstrained (lb = -Inf, ub = Inf)
        # (already initialized to -Inf/Inf)
        
        param_idx += hazard.npar_total
    end
    
    return lb, ub
end

"""
    _get_baseline_lb(family::Symbol, n_baseline::Int) -> Vector{Float64}

Get lower bounds for baseline parameters by hazard family.
"""
function _get_baseline_lb(family::Symbol, n_baseline::Int)
    if family == :exp
        # Exponential: rate ≥ 0
        return fill(NONNEG_LB, n_baseline)
    elseif family == :wei
        # Weibull: shape ≥ 0, scale ≥ 0
        return fill(NONNEG_LB, n_baseline)
    elseif family == :gom
        # Gompertz: shape ∈ ℝ (can be negative), rate ≥ 0
        return [n_baseline == 2 ? -Inf : NONNEG_LB, NONNEG_LB][1:n_baseline]
    elseif family == :sp
        # Spline: all coefficients ≥ 0 (non-negativity of hazard)
        return fill(NONNEG_LB, n_baseline)
    elseif family == :pt
        # Phase-type: rates ≥ 0
        return fill(NONNEG_LB, n_baseline)
    else
        @warn "Unknown hazard family :$family, using unconstrained bounds"
        return fill(-Inf, n_baseline)
    end
end

"""
    _resolve_user_bounds(user_bounds::Dict, model) -> (lb_vec, ub_vec)

Convert Dict{Symbol, NamedTuple} user bounds to flat vectors.
Keys are parameter names (e.g., :h12_rate, :h12_x).
Values are NamedTuples with optional :lb and :ub fields.
"""
function _resolve_user_bounds(user_bounds::Dict, model)
    n_params = length(model.parameters.flat)
    lb_vec = fill(-Inf, n_params)  # Default: no lower constraint
    ub_vec = fill(Inf, n_params)   # Default: no upper constraint
    
    # Build parameter name -> index mapping
    parnames = _get_flat_parnames(model)
    name_to_idx = Dict(name => i for (i, name) in enumerate(parnames))
    
    for (name, bounds) in user_bounds
        if !haskey(name_to_idx, name)
            throw(ArgumentError("Unknown parameter name :$name in user_bounds. " *
                               "Valid names: $(parnames)"))
        end
        idx = name_to_idx[name]
        
        # Extract lb and ub from NamedTuple if present
        if haskey(bounds, :lb)
            lb_vec[idx] = bounds.lb
        end
        if haskey(bounds, :ub)
            ub_vec[idx] = bounds.ub
        end
    end
    
    return lb_vec, ub_vec
end

"""
    _get_flat_parnames(model::MultistateProcess) -> Vector{Symbol}

Get parameter names in flat vector order.
"""
function _get_flat_parnames(model::MultistateProcess)
    return reduce(vcat, [h.parnames for h in model.hazards])
end

# Main API function - see Section 2.4 for full docstring
function generate_parameter_bounds(model::MultistateProcess; 
                                   user_bounds=nothing)
    # 1. Generate package bounds from hazard families (natural scale only)
    pkg_lb, pkg_ub = _generate_package_bounds(model)
    
    # 2. If no user bounds, return package bounds
    if isnothing(user_bounds)
        return pkg_lb, pkg_ub
    end
    
    # 3. Convert user bounds Dict to vectors
    user_lb_vec, user_ub_vec = _resolve_user_bounds(user_bounds, model)
    
    # 4. Combine via intersection
    final_lb = max.(pkg_lb, user_lb_vec)
    final_ub = min.(pkg_ub, user_ub_vec)
    
    conflicts = findall(final_lb .> final_ub)
    if !isempty(conflicts)
        parnames = _get_flat_parnames(model)
        throw(ArgumentError(
            "User bounds conflict with package constraints for: $(parnames[conflicts])"
        ))
    end
    
    return final_lb, final_ub
end
```

#### Step 1.2: Add to MultistateModels.jl module
**File**: `src/MultistateModels.jl`

```julia
include("utilities/bounds.jl")
export generate_parameter_bounds
```

#### Step 1.3: Remove estimation scale from bounds.jl
**File**: `src/utilities/bounds.jl`

**REMOVE** (not simplify - remove entirely):
- `scale::Symbol=:estimation` argument from `generate_parameter_bounds`
- `_transform_bounds_to_estimation` function
- `_get_positive_mask` function  
- All references to "estimation scale" in docstrings

**CHANGE**:
- Rename `POSITIVE_LB = 1e-10` to `NONNEG_LB = 0.0`
- Update all docstrings to say "non-negativity" instead of "positivity"

Parameters are now always on natural scale. Box constraints handle non-negativity directly.

### Phase 2: Spline Parameter Handling

#### Step 2.1: Simplify _spline_ests2coefs (remove exp)
**File**: `src/construction/multistatemodel.jl`

```julia
function _spline_ests2coefs(ests::AbstractVector{T}, basis, monotone::Int) where T
    if monotone == 0
        return ests  # SIMPLIFIED: Direct identity, no exp()
    else
        # For monotone: increments are already non-negative (box constrained)
        increments = ests  # SIMPLIFIED: No exp()
        # Build I-spline coefficients from increments
        ...
    end
end
```

#### Step 2.2: Update spline hazard function generation
**File**: `src/construction/multistatemodel.jl`

No `constrained` flag needed. The generated hazard functions call `_spline_ests2coefs` which now always expects natural-scale parameters. No changes needed here.

### Phase 3: Penalty Infrastructure (SIMPLIFY)

#### Step 3.1: Remove exp_transform from PenaltyTerm
**File**: `src/types/infrastructure.jl`

```julia
struct PenaltyTerm
    ...
    # REMOVE: exp_transform::Bool  # No longer needed
end
```

#### Step 3.2: Simplify compute_penalty
**File**: `src/types/infrastructure.jl`

**REMOVE** the exp_transform conditional entirely:
```julia
# Before (current):
for term in config.terms
    θ_j = @view parameters[term.hazard_indices]
    β_j = term.exp_transform ? exp.(θ_j) : θ_j  # REMOVE this conditional
    penalty += term.lambda * dot(β_j, term.S * β_j)
end

# After (simplified - parameters already on natural scale):
for term in config.terms
    β_j = @view parameters[term.hazard_indices]  # Direct use, no transform
    penalty += term.lambda * dot(β_j, term.S * β_j)
end
```

Penalty is now truly quadratic: P(β) = (λ/2) βᵀSβ

### Phase 4: Fitting Functions

#### Step 4.1: Update fit_penalized_beta
**File**: `src/inference/smoothing_selection.jl`

```julia
function fit_penalized_beta(model, data, lambda, penalty_config, beta_init;
                            maxiters=100, ...)
    
    lb, ub = generate_parameter_bounds(model)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=lb, ub=ub)
    ...
end
```

#### Step 4.2: Update _fit_exact
**File**: `src/inference/fit_exact.jl`

Add bounds passing to OptimizationProblem:
```julia
lb, ub = generate_parameter_bounds(model)
prob = OptimizationProblem(optf, parameters, data; lb=lb, ub=ub)
```

#### Step 4.3: Update _fit_markov
**File**: `src/inference/fit_markov.jl`

Similar updates for Markov model fitting.

#### Step 4.4: Configure Ipopt for exact constraint satisfaction
**Files**: `src/inference/fit_exact.jl`, `src/inference/smoothing_selection.jl`

By default, Ipopt relaxes bounds internally via `bound_relax_factor`. To ensure the final solution exactly satisfies box constraints:

```julia
# In IpoptOptimizer() calls, add:
sol = solve(prob, Ipopt.Optimizer();
    honor_original_bounds = "yes",  # Ensures β ≥ 0 exactly, not β ≥ -ε
    # ... other options
)
```

**Locations to update:**
- `_fit_exact()` in `fit_exact.jl` (lines ~135, ~189, ~256)
- `fit_penalized_beta()` in `smoothing_selection.jl` (line ~186)
- `select_smoothing_parameters()` in `smoothing_selection.jl`
- Any other `Ipopt.Optimizer()` calls

### Phase 5: PIJCV/Smoothing Selection

#### Step 5.1: Update SmoothingSelectionState
**File**: `src/inference/smoothing_selection.jl`

Ensure state is constructed with natural-scale parameters (now the only scale).

#### Step 5.2: Verify PIJCV criterion is quadratic
**File**: `src/inference/smoothing_selection.jl`

With parameters on natural scale and no exp_transform:
- P(β) = (λ/2) βᵀSβ ✓ (quadratic)
- Newton approximation valid ✓

#### Step 5.3: Update select_smoothing_parameters
**File**: `src/inference/smoothing_selection.jl`

No special flags needed - all code paths now use natural scale.

### Phase 6: Model Construction

#### Step 6.1: Update multistatemodel for natural-scale initialization
**File**: `src/construction/multistatemodel.jl`

When starting values provided:
- Before: Convert to log scale for storage
- After: Store directly on natural scale (validate non-negative for constrained params)

#### Step 6.2: Update parameter initialization
**File**: Various

Ensure `initialize_parameters` returns natural-scale values (now the only scale).

---

## Part 4: Test Modifications

### Tests That Must Change

| Test File | Reason | Changes |
|-----------|--------|---------|
| `unit/hazards.jl` | Parameter scale assumptions | Update expected values to natural scale |
| `unit/parameters.jl` | If exists, scale transformation tests | Update for identity transforms |
| `unit/spline_hazards.jl` | Spline coefficient tests | Update for constrained mode |
| `integration/fit_exact_*.jl` | Fitting with bounds | Verify bounds respected |
| `longtests/spline_*.jl` | Spline inference tests | Ensure still converge |

### New Tests Required

1. **Bounds generation**: Test `generate_parameter_bounds` produces correct bounds by family
2. **Constrained optimization**: Test optimizer respects lb/ub
3. **PIJCV with constraints**: Test λ selection is now reasonable
4. **User/Package constraint combination**: Test intersection logic

### Constraint Combination Test Cases

```julia
@testset "User/Package Constraint Combination" begin
    # Setup: model with spline hazard (requires non-negative coefficients)
    h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; degree=3)
    model = multistatemodel(h12; data=test_data)
    calibrate_splines!(model; nknots=5)
    
    @testset "Package bounds only" begin
        lb, ub = generate_parameter_bounds(model)
        # Spline coefs should have non-negative lb
        @test all(lb[1:h12.npar_baseline] .>= 0)
        # Covariate coefs should be unconstrained
        @test lb[end] == -Inf
        @test ub[end] == Inf
    end
    
    @testset "User tightens bounds (allowed)" begin
        user_bounds = Dict(:h12_x => (lb=-5.0, ub=5.0))
        lb, ub = generate_parameter_bounds(model; user_bounds)
        # User bound should apply to covariate
        @test lb[end] == -5.0
        @test ub[end] == 5.0
        # Package bounds still apply to spline coefs
        @test all(lb[1:h12.npar_baseline] .>= 0)
    end
    
    @testset "User tries to loosen package bounds (blocked)" begin
        # Try to make spline coef unconstrained (should be blocked)
        # Get a spline param name
        spline_param = h12.parnames[1]
        user_bounds = Dict(spline_param => (lb=-Inf,))
        lb, ub = generate_parameter_bounds(model; user_bounds)
        # Package lb should win for non-negative-constrained params
        @test all(lb[1:h12.npar_baseline] .>= 0)
    end
    
    @testset "Conflicting bounds error" begin
        # User lb > user ub should error
        user_bounds = Dict(:h12_x => (lb=10.0, ub=5.0))
        @test_throws ArgumentError generate_parameter_bounds(model; user_bounds)
    end
    
    @testset "Unknown parameter name error" begin
        user_bounds = Dict(:nonexistent_param => (lb=0.0,))
        @test_throws ArgumentError generate_parameter_bounds(model; user_bounds)
    end
    
    @testset "User specifies only lb or only ub" begin
        # Only lower bound
        user_bounds = Dict(:h12_x => (lb=-5.0,))
        lb, ub = generate_parameter_bounds(model; user_bounds)
        @test lb[end] == -5.0
        @test ub[end] == Inf  # Default upper bound preserved
        
        # Only upper bound  
        user_bounds = Dict(:h12_x => (ub=5.0,))
        lb, ub = generate_parameter_bounds(model; user_bounds)
        @test lb[end] == -Inf  # Default lower bound preserved
        @test ub[end] == 5.0
    end
end
```

---

## Part 5: Risks and Mitigations

### Risk 1: Performance regression
- **Issue**: Box constraints may slow optimization
- **Mitigation**: Ipopt handles box constraints natively, minimal overhead
- **Test**: Benchmark before/after on standard test cases

### Risk 2: Breaking existing tests
- **Issue**: 1483 tests currently pass with log-scale params
- **Mitigation**: This is a breaking change - tests will need to be updated. No backward compatibility mode.
- **Test**: Run full test suite after each phase, update as needed

### Risk 3: Numerical instability near bounds
- **Issue**: Parameters hitting lower bounds (0) may cause issues
- **Mitigation**: Use `honor_original_bounds="yes"` in Ipopt; PIJCV projects via `max.(beta_loo, 0.0)`
- **Test**: Add tests for edge cases near bounds

### Risk 4: AD compatibility with bounds
- **Issue**: ForwardDiff needs to work at boundary
- **Mitigation**: Ipopt handles this; test gradient computation at bounds
- **Test**: Verify AD works correctly with constrained optimization

---

## Part 6: Validation Checklist

### Per-Phase Validation

- [ ] Phase 1: `generate_parameter_bounds` returns correct bounds for all families
- [ ] Phase 2: `_spline_ests2coefs(x, b, 0) == x` (identity, no exp)
- [ ] Phase 3: `compute_penalty` gives P = λβᵀSβ/2 (no exp_transform logic)
- [ ] Phase 4: `fit_penalized_beta` respects bounds (params ≥ lb)
- [ ] Phase 5: PIJCV selects reasonable λ (not extreme values)
- [ ] Phase 6: Model construction stores natural-scale params (only scale)

### Final Validation

- [ ] Tests updated and passing
- [ ] PIJCV selects λ similar to PERF/EFS/exact k-fold
- [ ] Penalized spline inference produces reasonable curves
- [ ] No performance regression in fitting speed

---

## Part 7: Implementation Timeline

| Phase | Est. Time | Dependencies |
|-------|-----------|--------------|
| Phase 1 (Bounds) | 2 hours | None |
| Phase 2 (Splines) | 2 hours | Phase 1 |
| Phase 3 (Penalty) | 1 hour | Phase 1 |
| Phase 4 (Fitting) | 3 hours | Phases 1-3 |
| Phase 5 (PIJCV) | 2 hours | Phase 4 |
| Phase 6 (Construction) | 2 hours | Phases 1-2 |
| Testing | 4 hours | All phases |
| **Total** | **16 hours** | |

---

## Appendix A: Critical Code Snippets

### A.1 Current _fit_exact optimization setup (fit_exact.jl:88-95)
```julia
adtype = DifferentiationInterface.SecondOrder(
    Optimization.AutoForwardDiff(), 
    Optimization.AutoForwardDiff()
)
optf = OptimizationFunction(make_objective(model, samplepaths), adtype)
prob = OptimizationProblem(optf, parameters, data)  # No lb/ub!
sol = solve(prob, optimizer; ...)
```

### A.2 Current fit_penalized_beta (smoothing_selection.jl:184-186)
```julia
optf = OptimizationFunction(penalized_nll, adtype)
prob = OptimizationProblem(optf, beta_init, nothing)  # No lb/ub!
sol = solve(prob, IpoptOptimizer(); ...)
```

### A.3 Current compute_penalty exp transform (infrastructure.jl:550) - TO BE REMOVED
```julia
# BEFORE (current - problematic):
β_j = term.exp_transform ? exp.(θ_j) : θ_j
penalty += lambda[lambda_idx] * dot(β_j, term.S * β_j)

# AFTER (simplified - no exp_transform):
penalty += lambda[lambda_idx] * dot(θ_j, term.S * θ_j)
```

---

## Appendix B: Reference Material

- **Wood (2024)**: "On Neighbourhood Cross Validation" arXiv:2404.16490v4
- **Optimization.jl docs**: Box constraints via `lb`, `ub` keyword arguments
- **Ipopt manual**: Native support for variable bounds
- **Project handoff**: `PENALIZED_SPLINES_HANDOFF_20260106.md`

---

## Part 8: Critical Mathematical Review (2025-01-06)

This section documents a rigorous mathematical review of the implementation plan, identifying gaps, unstated assumptions, and potential failure modes.

### 8.1 Fundamental Mathematical Concerns

#### Issue 1: Barrier Function Interaction (CRITICAL)

**Status:** RESOLVED - Not an issue

**The Problem:**
When using interior point methods (Ipopt), the actual objective being optimized is:

$$L_\mu(\beta) = -\ell(\beta) + P(\beta) - \mu \sum_{i \in \mathcal{P}} \log(\beta_i - \varepsilon)$$

where $\mathcal{P}$ is the set of non-negativity-constrained parameters and μ is the barrier parameter.

The Hessian becomes:

$$H_{L_\mu} = H_\ell + \lambda S + \mu \cdot \text{diag}\left(\frac{\mathbf{1}_{\mathcal{P}}}{(\beta - \varepsilon)^2}\right)$$

The PIJCV Newton approximation uses $H_\lambda = H_\ell + \lambda S$, NOT $H_{L_\mu}$.

**Resolution:** After code review, **this is not an issue**. The PIJCV implementation computes its own Hessian independently of Ipopt:
1. `H_unpenalized` is computed via AD on the log-likelihood (sum of subject Hessians)
2. `H_lambda = H_unpenalized + penalty_Hessian` via `_build_penalized_hessian`
3. The barrier function never enters this computation

The PIJCV Hessian is correct by construction.

---

#### Issue 2: Active Constraints and Newton Approximation (LOW)

**Status:** ✅ RESOLVED

**The Problem:**
If $\hat{\beta}_i = 0$ (constraint active), the Newton LOO approximation:

$$\hat{\beta}^{(-j)} \approx \hat{\beta} + H_\lambda^{-1} g_j$$

can yield $\hat{\beta}^{(-j)}_i < 0$, violating the non-negativity constraint.

**Scenarios where constraints become active:**
1. Spline coefficient at boundary → hazard approaches zero (more common for monotone splines)
2. Rate parameter at boundary → very slow transitions
3. Occasionally occurs for non-monotone splines as well

**Resolution (two-part):**

1. **Ipopt setting:** Use `honor_original_bounds = "yes"` in Ipopt configuration to ensure final MLE solution respects box constraints exactly (Ipopt relaxes bounds internally by default via `bound_relax_factor`).

2. **PIJCV LOO projection:** In the Newton LOO approximation, project back to feasible region:
   ```julia
   beta_loo = beta_hat + H_lambda_inv * g_j
   beta_loo_proj = max.(beta_loo, 0.0)  # Non-negativity, not positivity
   ```

**Rationale:**
- Newton LOO is an approximation; small constraint violations are within approximation error
- Projection is simple, cheap, and maintains feasibility for the PIJCV criterion
- No warnings needed—don't over-engineer
- Project to 0, not ε: this is a non-negativity constraint, not positivity

---

#### Issue 3: Penalty Matrix Null Space (LOW)

**Status:** ✅ RESOLVED - Not a concern

**The Problem:**
For difference penalties, $S = D_m^\top D_m$ where $D_m$ is an m-th order difference matrix. This S has:
- Rank = K - m (K = number of basis functions)
- Null space = polynomials of degree < m (level, slope for m=2)

**Resolution:** Do nothing. This is standard P-spline behavior.

**Rationale:**
1. The null space contains level and slope — exactly what the data constrains most strongly
2. mgcv uses rank-deficient penalties without ridge augmentation
3. Unstable estimates of level/slope would indicate a data problem, not a methodology problem
4. Box constraints ($\beta_i \geq 0$) provide additional implicit regularization

---

### 8.2 Numerical Analysis Concerns

#### Issue 4: Choice of Lower Bound (LOW)

**Status:** ✅ RESOLVED

**Resolution:** Use ε = 0 (non-negativity constraint, not positivity).

**Rationale:**
- If the optimizer wants β ≈ 0, that's statistically meaningful (near-zero hazard)
- The data will naturally prevent degenerate solutions
- If numerical issues arise, can tighten to 1e-8 later

---

#### Issue 5: Gradient Scaling Heterogeneity (LOW)

**Status:** ✅ RESOLVED - Non-issue

**Resolution:** Accept Ipopt's internal scaling. No action needed.

**Rationale:**
- Ipopt has built-in gradient-based scaling (`nlp_scaling_method = "gradient-based"` is default)
- This is standard for constrained optimization — Ipopt is designed to handle it
- We're not using a hand-rolled Newton method

---

### 8.3 Statistical Concerns

#### Issue 6: Monotone Splines Penalty Transform (MEDIUM)

**Status:** ✅ RESOLVED

**The Problem:**
For monotone hazards, the constraint set {β : β₁ ≤ β₂ ≤ ... ≤ β_K, β₁ ≥ 0} is a polyhedral cone, not a box.

**Resolution:**

Keep the current parameterization where monotone splines optimize over **increments** γ (with β = Lγ where L is the cumsum matrix):

1. **Non-monotone (`monotone=0`):**
   - Optimize coefficients β directly
   - Box constraints: β_i ≥ 0
   - Penalty matrix: standard S
   - No transformation needed

2. **Monotone (`monotone≠0`):**
   - Optimize increments γ directly
   - Box constraints: γ_i ≥ 0 (ensures monotonicity)
   - Penalty matrix: **transformed** S̃ = LᵀSL
   - Penalty: γᵀS̃γ = βᵀSβ (same smoothness as non-monotone)

---

##### Monotone Increasing (`monotone = 1`)

The transformation from increments γ to coefficients β (from `_spline_ests2coefs`):

$$\beta_1 = \gamma_1$$
$$\beta_i = \gamma_1 + \sum_{j=2}^{i} \gamma_j \cdot w_j \quad \text{for } i \geq 2$$

where $w_j = \frac{t_{j+k} - t_j}{k}$ are the I-spline weights.

**Matrix form:** $\beta = L_+ \gamma$ where:

$$L_+ = \begin{pmatrix} 
1 & 0 & 0 & 0 & \cdots \\
1 & w_2 & 0 & 0 & \cdots \\
1 & w_2 & w_3 & 0 & \cdots \\
1 & w_2 & w_3 & w_4 & \cdots \\
\vdots & & & & \ddots
\end{pmatrix}$$

First column is all 1s (intercept), subsequent columns have weights that accumulate down.

**Transformed penalty:** $\tilde{S}_+ = L_+^\top S L_+$

---

##### Monotone Decreasing (`monotone = -1`)

The code computes the increasing transformation, then reverses:

$$\beta^{(inc)} = L_+ \gamma, \quad \beta = \text{reverse}(\beta^{(inc)}) = R \cdot L_+ \gamma$$

where $R$ is the reversal (anti-diagonal) matrix.

**Matrix form:** $L_- = R \cdot L_+$

**Transformed penalty:**
$$\tilde{S}_- = L_-^\top S L_- = (R L_+)^\top S (R L_+) = L_+^\top R S R L_+$$

Since $R^\top = R$ and $RSR$ is $S$ with rows and columns reversed.

---

##### Implementation

```julia
"""
    build_monotone_cumsum_matrix(basis, monotone::Int) -> Matrix{Float64}

Build the cumulative sum transformation matrix L such that β = Lγ.
"""
function build_monotone_cumsum_matrix(basis, monotone::Int)
    n = BSplineKit.length(basis)
    
    if monotone == 0
        return Matrix{Float64}(I, n, n)
    end
    
    k = BSplineKit.order(basis)
    t = BSplineKit.knots(basis)
    
    # Build L for increasing case
    L = zeros(Float64, n, n)
    
    # First column: all 1s (intercept contribution)
    L[:, 1] .= 1.0
    
    # Subsequent columns: accumulated weights
    for j in 2:n
        w_j = (t[j + k] - t[j]) / k
        for i in j:n
            L[i, j] = w_j
        end
    end
    
    if monotone == -1
        # Apply reversal: L_decreasing = R * L_increasing
        L = L[end:-1:1, :]
    end
    
    return L
end

"""
    transform_penalty_matrix_for_monotone(S, basis, monotone) -> Matrix{Float64}

Transform penalty matrix so penalty on increments γ equals penalty on coefficients β.
Returns S̃ = LᵀSL. For monotone=0, returns S unchanged.
"""
function transform_penalty_matrix_for_monotone(S::Matrix{Float64}, basis, monotone::Int)
    if monotone == 0
        return S
    end
    L = build_monotone_cumsum_matrix(basis, monotone)
    return L' * S * L
end
```

---

#### Issue 7: Weibull Shape Parameter Near Zero (LOW)

**Status:** ✅ RESOLVED

**Resolution:** Use κ ≥ 0 (non-negativity). No special handling needed.

**Rationale:**
- The data will prevent κ → 0 — you can't fit κ ≈ 0 unless the data supports it
- If κ → 0 is the MLE, something is wrong with the model specification, not the optimization
- This is equally true under log-transform (current implementation)

---

### 8.4 Implementation Architecture Concerns

#### Issue 8: Gompertz Shape Genuinely Unconstrained (ACCEPTABLE)

**Status:** RESOLVED - Accept heterogeneity

Gompertz shape can be negative (decreasing hazard), zero (constant), or positive (increasing). This is **intentional** and **meaningful**.

**Resolution:** Accept that different families have different constraint structures. Document clearly that "natural scale" means different things for different parameters.

---

#### Issue 9: Parameter Storage Architecture (MEDIUM)

**Status:** TODO - Implement during Phase 2

**The Problem:**
Under Strategy B, `parameters.flat` and `parameters.natural` become redundant for spline hazards since both are on natural scale.

**Resolution:** Remove the redundancy. This is an opportunity to simplify the codebase.

**Implementation plan:**
1. Audit all uses of `parameters.flat` vs `parameters.natural`
2. Consolidate to a single representation where possible
3. Keep clear documentation of what scale each parameter family uses
4. Ensure user-facing functions always return natural-scale parameters

**Benefits:**
- Simpler, more maintainable code
- Fewer places for bugs to hide
- Clearer mental model for developers

---

### 8.5 Testing and Validation Gaps

#### Issue 10: PIJCV Validation Criteria (MEDIUM)

**Status:** DEFERRED - Focus on parameter handling first

**The Problem:**
The plan's validation criterion "PIJCV selects λ similar to PERF/EFS/exact k-fold" is flawed because these methods optimize **different objectives**.

**Better validation (for later):**
1. Compare PIJCV to **exact LOOCV** (same criterion, different computation)
2. Report **EDF** at selected λ, not just λ value
3. Report **prediction error** on held-out data
4. Visual inspection of fitted hazards

**Note:** Will revisit after parameter handling is complete.

---

#### Issue 11: Missing Edge Case Tests (MEDIUM)

**Status:** TODO - Add to test plan

**Edge cases to test:**
1. **Active constraints:** Simulate data where one spline region has near-zero hazard
2. **Large λ:** Verify convergence with heavy smoothing
3. **Small λ:** Verify convergence with minimal smoothing  
4. **Monotone splines:** Test increasing and decreasing hazards
5. **Competing risks:** Test shared λ and total hazard penalty
6. **Sparse data:** Few events per transition
7. **Zero transitions:** Pathway with no observed events
8. **Extreme covariates:** Large covariate values

**Implementation:** Add these to test suite during implementation phase.

---

### 8.6 Summary Table of Issues

| # | Issue | Severity | Status | Section |
|---|-------|----------|--------|---------|
| 1 | Barrier function impact on PIJCV Hessian | N/A | ✅ RESOLVED | 8.1 |
| 2 | Active constraints in Newton LOO | LOW | ✅ RESOLVED | 8.1 |
| 3 | Penalty null space handling | LOW | ✅ RESOLVED | 8.1 |
| 4 | Lower bound choice | LOW | ✅ RESOLVED | 8.2 |
| 5 | Gradient scaling heterogeneity | LOW | ✅ RESOLVED | 8.2 |
| 6 | Monotone splines penalty transform | MEDIUM | ✅ RESOLVED | 8.3 |
| 7 | Weibull shape near zero | LOW | ✅ RESOLVED | 8.3 |
| 8 | Gompertz shape unconstrained | N/A | ✅ RESOLVED | 8.4 |
| 9 | Parameter storage architecture | MEDIUM | 🔧 TODO | 8.4 |
| 10 | PIJCV validation criteria | MEDIUM | ⏸️ DEFERRED | 8.5 |
| 11 | Missing edge case tests | MEDIUM | 🔧 TODO | 8.5 |

### 8.7 Summary of Resolutions

**Resolved (no action needed):**
- Issue 1: N/A — PIJCV computes its own Hessian via AD
- Issue 2: Projected Newton + `honor_original_bounds="yes"` in Ipopt
- Issue 3: Standard P-spline behavior, likelihood constrains level/slope
- Issue 4: Use ε = 0 (non-negativity)
- Issue 5: Non-issue, Ipopt handles gradient scaling
- Issue 6: Transform penalty matrix S̃ = LᵀSL for monotone splines
- Issue 7: Use κ ≥ 0 (non-negativity)
- Issue 8: Gompertz shape correctly unconstrained by design

**TODO during implementation:**
- Issue 9: Remove parameter storage redundancy (simplify codebase)
- Issue 11: Add edge case tests

**Deferred:**
- Issue 10: PIJCV validation — revisit after parameter handling complete

---

## Part 9: Comprehensive Change List (Audit-Verified)

This section provides the definitive list of ALL changes needed, verified by codebase audit.

### 9.1 Phase 1: bounds.jl Changes

**File**: `src/utilities/bounds.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Rename constant | ~24 | `POSITIVE_LB = 1e-10` → `NONNEG_LB = 0.0` |
| Remove argument | ~34 | Remove `scale::Symbol=:estimation` from `generate_parameter_bounds` |
| Remove function | ~187-232 | Delete `_transform_bounds_to_estimation` |
| Remove function | ~234-252 | Delete `_get_positive_mask` |
| Remove function | ~254-296 | Delete `_estimation_scale_lb` |
| Remove function | ~298-340 | Delete `_estimation_scale_ub` |
| Update docstrings | Throughout | Remove "estimation scale" references |
| Update return | ~90 | Return natural scale bounds only (remove scale conditional) |

### 9.2 Phase 2: Spline Handling Changes

**File**: `src/construction/multistatemodel.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Simplify transform | 669-695 | `_spline_ests2coefs`: Remove `exp.()` calls, use identity for monotone==0 |
| Update inverse | 706-750 | `_spline_coefs2ests`: Remove `log.()` calls, use identity for monotone==0 |

**Current code at line 672**:
```julia
if monotone == 0
    return exp.(ests)  # REMOVE exp
```

**New code**:
```julia
if monotone == 0
    return ests  # Direct identity (box constraints ensure non-negativity)
```

### 9.3 Phase 3: Penalty Infrastructure Changes

**File**: `src/types/infrastructure.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Remove field | 404-438 | `struct PenaltyTerm`: Remove `exp_transform::Bool` field |
| Remove constructor | 411-426 | Remove both constructors, replace with single 5-arg constructor |
| Simplify compute_penalty | 538-570 | Remove `exp_transform` conditional |

**File**: `src/inference/smoothing_selection.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Remove docstring refs | 67-76 | Remove "estimation scale" and "exp_transform" from docstrings |
| Simplify penalty | 88 | `β_j = term.exp_transform ? exp.(θ_j) : θ_j` → `β_j = θ_j` |
| Simplify total haz | 99 | `β_total .+= exp.(θ_k)` → `β_total .+= θ_k` |
| Update config copy | 1904-1912 | `update_penalty_config_lambda`: Remove `term.exp_transform` arg |
| Simplify Hessian | 996-1010 | `_build_penalized_hessian`: Remove `exp_transform` conditional |
| Add PIJCV projection | 374 | After `beta_loo = state.beta_hat .+ delta_i`, add `beta_loo = max.(beta_loo, 0.0)` |

### 9.4 Phase 4: Fitting Function Changes

**File**: `src/inference/fit_common.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Add Ipopt option | 100-107 | In `_solve_optimization`: Add `honor_original_bounds="yes"` for Ipopt |

**Current code**:
```julia
if _is_ipopt_solver(solver)
    return solve(prob, _solver; print_level = 0)
```

**New code**:
```julia
if _is_ipopt_solver(solver)
    return solve(prob, _solver; print_level = 0, honor_original_bounds = "yes")
```

**File**: `src/inference/smoothing_selection.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Add bounds to prob | 186-220 | `fit_penalized_beta`: Call `generate_parameter_bounds`, add lb/ub to OptimizationProblem |
| Add Ipopt option | 203-210 | Add `honor_original_bounds="yes"` to direct `solve()` calls |

### 9.5 Phase 5: Penalty Config Construction

**File**: `src/utilities/penalty_config.jl`

| Change | Lines | Description |
|--------|-------|-------------|
| Update constructors | 215-240 | `PenaltyTerm(...)` calls: Remove 6th arg (exp_transform) |

**Note**: Current code uses 5-arg constructor (backward compat), which defaults `exp_transform=true`. After removing the field, these work as-is.

### 9.6 Files That Need NO Changes

- `src/hazard/generators.jl` — Already expects natural scale params
- `src/hazard/spline.jl` — Calls `_spline_ests2coefs` (which we're fixing)
- `src/utilities/transforms.jl` — Consider deprecating, but low priority
- `src/output/accessors.jl` — `:estimation` scale aliases `:flat`, keep for backward compat

### 9.7 Test File Changes

**File**: `test/unit/test_bounds.jl` (REWRITE)

| Change | Description |
|--------|-------------|
| Remove estimation scale tests | Tests for `scale=:estimation` |
| Add Dict API tests | Tests for `user_bounds` Dict format |
| Add conflict tests | Tests for lb/ub conflict detection |
| Add per-family tests | Tests for correct lb by hazard family |

### 9.8 Implementation Order (Dependency-Safe)

```
Phase 1.1: bounds.jl — Remove estimation scale functions and update constant
    ↓
Phase 1.2: test_bounds.jl — Rewrite tests for Dict API and natural scale
    ↓
Phase 2: _spline_ests2coefs — Remove exp() transform
    ↓
Phase 3.1: PenaltyTerm struct — Remove exp_transform field
    ↓
Phase 3.2: compute_penalty — Remove exp_transform conditional
    ↓
Phase 3.3: smoothing_selection.jl — Remove exp_transform references
    ↓
Phase 4.1: fit_common.jl — Add honor_original_bounds to Ipopt
    ↓
Phase 4.2: fit_penalized_beta — Add lb/ub to OptimizationProblem
    ↓
Phase 5: PIJCV — Add max(beta_loo, 0) projection
    ↓
Phase 6: Run full test suite, fix regressions
```

---

## Next Steps

1. **Start with Phase 1** — Fix bounds.jl (remove estimation scale, change lb to 0.0)
2. **Rewrite test_bounds.jl** — Update tests for new API
3. **Proceed through phases** in dependency order
4. **Run tests after each phase** to catch regressions early
