# Part 3: AD Compatibility Analysis

## 3.1 Current AD Strategy

### Supported Backends

| Backend | Mode | Status | Primary Use |
|---------|------|--------|-------------|
| ForwardDiff | Forward | ✅ Full support | Default for optimization |
| Mooncake | Reverse | ⚠️ Partial | Future primary |
| Enzyme | Reverse | ❌ Not tested | Future option |

### AD Selection in Code

**Location**: `modelfitting.jl:50-80`

```julia
abstract type ADBackend end
struct ForwardDiffBackend <: ADBackend end
struct MooncakeBackend <: ADBackend end
struct EnzymeBackend <: ADBackend end

function get_optimization_ad(backend::ForwardDiffBackend)
    return Optimization.AutoForwardDiff()
end

function get_optimization_ad(backend::MooncakeBackend)
    return Optimization.AutoMooncake()
end

function get_optimization_ad(backend::EnzymeBackend)
    return Optimization.AutoEnzyme()
end
```

---

## 3.2 ForwardDiff Compatibility

### How ForwardDiff Works

ForwardDiff uses **Dual numbers** for automatic differentiation:
```julia
struct Dual{T,V,N} <: Real
    value::V
    partials::Partials{N,V}
end
```

When you call `f(Dual(x, 1.0))`, ForwardDiff propagates derivatives through arithmetic operations.

### Requirements for ForwardDiff Compatibility

1. **Type-generic code**: Functions must accept `AbstractVector{<:Real}` not just `Vector{Float64}`
2. **Preserve Dual types**: Don't convert to `Float64` mid-computation
3. **Mutation is OK**: ForwardDiff handles mutating operations

### Current ForwardDiff Support

**Parameter unflatten** (helpers.jl):
```julia
function unflatten_parameters(flat_params::AbstractVector{T}, model) where {T<:Real}
    if T === Float64
        params = unflatten(model.parameters.reconstructor, flat_params)
    else
        # Use AD-compatible unflatten for Dual types
        params = unflattenAD(model.parameters.reconstructor, flat_params)
    end
    return to_natural_scale(params, model.hazards, T)
end
```

**Likelihood functions** accept generic element types:
```julia
function loglik_markov(parameters, data::MPanelData; neg=true)
    T = eltype(parameters)  # Float64 or Dual
    ll = zero(T)
    # ... computations preserve T
end
```

### ForwardDiff Issues Found

1. **Type conversion in hazard eval**:
```julia
# PROBLEM: Forces Float64
rate = Float64(pars.baseline[1])

# SOLUTION: Keep generic
rate = pars.baseline[1]
```

2. **Pre-allocated containers with wrong type**:
```julia
# PROBLEM: Pre-allocated as Float64
hazmat_book = build_hazmat_book(Float64, tmat, books[1])

# SOLUTION: Use parameter element type
hazmat_book = build_hazmat_book(eltype(parameters), tmat, books[1])
```

---

## 3.3 Reverse-Mode AD Compatibility (Mooncake/Enzyme)

### How Reverse-Mode AD Works

Reverse-mode AD:
1. **Forward pass**: Record operations to a "tape"
2. **Backward pass**: Propagate gradients from output to inputs

Advantages:
- O(1) cost for gradients w.r.t. many parameters
- Better for large parameter vectors

### Requirements for Reverse-Mode Compatibility

1. **No mutation of inputs**: Can't modify arrays that gradients flow through
2. **No mutation of intermediates** (in some backends): Enzyme/Mooncake may struggle with in-place operations
3. **Avoid certain control flow**: Some backends have issues with complex branches

### Current Reverse-Mode Support

**Dual implementation pattern** (likelihoods.jl):

```julia
# Mutating version (ForwardDiff)
function loglik_markov(parameters, data::MPanelData; neg=true)
    pars = safe_unflatten(parameters, data.model)
    
    # MUTATION: Pre-allocated containers
    hazmat_book = build_hazmat_book(...)
    tpm_book = build_tpm_book(...)
    
    for t in eachindex(books[1])
        compute_hazmat!(hazmat_book[t], ...)  # In-place
        compute_tmat!(tpm_book[t], ...)       # In-place
    end
    # ...
end

# Non-mutating version (Mooncake/Enzyme)
function loglik_markov_functional(parameters, data::MPanelData; neg=true)
    pars = safe_unflatten(parameters, data.model)
    
    # NO MUTATION: Build TPMs functionally
    tpm_dict = Dict{Tuple{Int,Int}, Matrix{T}}()
    for (t_idx, tpm_index_df) in enumerate(data.books[1])
        Q = compute_hazmat(T, n_states, pars, hazards, tpm_index_df, data.model.data)
        for t in eachindex(tpm_index_df.tstop)
            dt = tpm_index_df.tstop[t]
            P = compute_tmat(Q, dt)  # Returns new matrix
            tpm_dict[(t_idx, t)] = P
        end
    end
    # ...
end
```

### Reverse-Mode Issues Found

1. **`exponential!` mutates in place**:
```julia
# PROBLEM: In-place mutation
exponential!(P, ExpMethodGeneric(), cache)

# SOLUTION: Functional version
P = exp(Q * dt)  # Or use non-mutating exponential
```

2. **Forward algorithm mutation**:
```julia
# PROBLEM: Mutates lmat
lmat[s, ind] += q[r, s] * lmat[r, ind - 1]

# SOLUTION: Build new array each step (slower but AD-compatible)
```

3. **FFBS mutation**:
```julia
# PROBLEM: BackwardSampling! writes to subj_dat
subj_dat.stateto[end] = rand(Categorical(vec(p)))

# NOTE: This is sampling, not likelihood - may not need AD
```

---

## 3.4 Unified AD Strategy

### Option A: Dual Implementation (Current)

Maintain two versions of critical functions:
- `loglik_markov` (mutating, ForwardDiff)
- `loglik_markov_functional` (non-mutating, Mooncake/Enzyme)

**Dispatch based on backend**:
```julia
function loglik_markov_dispatch(parameters, data; neg=true, backend=ForwardDiffBackend())
    if backend isa ForwardDiffBackend
        return loglik_markov(parameters, data; neg=neg)
    else
        return loglik_markov_functional(parameters, data; neg=neg)
    end
end
```

**Pros**: Maximum performance for each backend
**Cons**: Code duplication, maintenance burden

### Option B: Functional Only

Use non-mutating code everywhere:
```julia
function loglik_markov(parameters, data; neg=true)
    # Always use functional implementation
    # ForwardDiff works fine with this
    # Mooncake/Enzyme also work
end
```

**Pros**: Single implementation, simpler maintenance
**Cons**: May be 10-30% slower for ForwardDiff due to allocations

### Option C: Mutation Barriers

Use "checkpoints" that break the tape at strategic points:
```julia
function loglik_markov(parameters, data; neg=true)
    # Compute TPMs with mutation (not AD'd)
    tpm_book = Mooncake.@non_differentiable compute_all_tpms(parameters, data)
    
    # Likelihood accumulation (AD'd)
    ll = compute_likelihood_from_tpms(parameters, tpm_book, data)
    return ll
end
```

**Pros**: Keep mutations where safe, AD where needed
**Cons**: Requires careful analysis of what needs gradients

### Recommendation

1. **Profile both implementations** with ForwardDiff
2. **If <15% difference**: Use Option B (functional only)
3. **If >15% difference**: Use Option A with dispatch, or Option C with barriers

---

## 3.5 Functions Requiring AD Compatibility

### Must Support AD (Gradients Needed)

| Function | Used In | Current Status |
|----------|---------|----------------|
| `loglik_markov` | Panel data fitting | ✅ ForwardDiff, ⚠️ Mooncake via `_functional` |
| `loglik_semi_markov` | MCEM M-step | ✅ ForwardDiff only |
| `loglik_exact` | Exact data fitting | ✅ ForwardDiff only |
| `loglik_path` | Per-path likelihood | ✅ ForwardDiff only |
| `survprob` | Survival probability | ✅ ForwardDiff |
| `eval_hazard` | Hazard evaluation | ✅ ForwardDiff |
| `eval_cumhaz` | Cumulative hazard | ✅ ForwardDiff |

### Does NOT Need AD (Sampling/Setup)

| Function | Reason |
|----------|--------|
| `simulate_path` | Forward simulation, no gradients |
| `draw_samplepath` | FFBS sampling |
| `sample_ecctmc!` | CTMC sampling |
| `ForwardFiltering!` | Probability computation for sampling |
| `BackwardSampling!` | State sequence sampling |
| `build_hazmat_book` | Setup, not in gradient path |
| `build_tpm_mapping` | Data preprocessing |

---

## 3.6 Testing AD Compatibility

### Test Script Template

```julia
using MultistateModels
using ForwardDiff
using Mooncake
using Test

# Setup model
model = ... # Create test model

# Get parameters and data
params = get_parameters_flat(model)
data = MPanelData(model, books)

# Test ForwardDiff gradient
@testset "ForwardDiff" begin
    f(p) = loglik_markov(p, data; neg=true)
    grad_fd = ForwardDiff.gradient(f, params)
    @test all(isfinite.(grad_fd))
end

# Test Mooncake gradient
@testset "Mooncake" begin
    f(p) = loglik_markov_functional(p, data; neg=true)
    grad_mc = Mooncake.gradient(f, params)
    @test all(isfinite.(grad_mc))
    
    # Compare to ForwardDiff
    @test grad_mc ≈ grad_fd rtol=1e-6
end

# Test Hessian (ForwardDiff only typically)
@testset "Hessian" begin
    f(p) = loglik_markov(p, data; neg=true)
    hess = ForwardDiff.hessian(f, params)
    @test all(isfinite.(hess))
    @test issymmetric(hess)
end
```

---

## Action Items from AD Analysis

| Item | Priority | Effort |
|------|----------|--------|
| Benchmark `loglik_markov` vs `_functional` | High | Low |
| Add Mooncake gradient tests | High | Medium |
| Decide on unified strategy | High | Decision |
| Implement chosen strategy | Medium | Medium-High |
| Document AD requirements | Low | Low |
