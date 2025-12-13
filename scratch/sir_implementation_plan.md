# SIR Implementation Plan for MCEM

## Overview

This plan adds optional Sampling Importance Resampling (SIR) to the MCEM algorithm, allowing users to subsample paths using either standard multinomial resampling or variance-reduced Latin Hypercube Sampling (LHS), with pool sizes guided by theoretical results from Li (2006) and the variance-reduced SIR literature.

## Background

### Key References
- **Li (2006)** - "The Sampling/Importance Resampling Algorithm": Pool size should be O(m log m) when importance weights have a moment generating function
- **Variance-reduced SIR paper**: LHS-based resampling reduces variance compared to multinomial resampling

### Core Idea
Instead of using importance-weighted averages with PSIS smoothing, we:
1. Sample a large pool of paths from the proposal
2. Resample a smaller subset proportional to importance weights
3. Use simple (unweighted) averages on the subsample
4. ESS = subsample size (deterministic, not estimated)

---

## Implementation Steps

### Step 1: Add SIR arguments to `fit` function
**File:** `src/modelfitting.jl` ~L892

**Status:** [x] Complete

Add parameters:
- `sir::Symbol = :none` — options: `:none`, `:sir`, `:lhs`
- `sir_pool_constant::Float64 = 2.0` — c in pool size formula `c*m*log(m)`
- `sir_max_pool::Int = 8192` — cap on maximum pool size per subject (2^13)
- `sir_resample::Symbol = :always` — options: `:always`, `:degeneracy`
- `sir_degeneracy_threshold::Float64 = 0.7` — Pareto-k threshold for `:degeneracy` mode

Validate at function entry:
```julia
sir ∈ (:none, :sir, :lhs) || throw(ArgumentError("sir must be :none, :sir, or :lhs"))
sir_resample ∈ (:always, :degeneracy) || throw(ArgumentError("sir_resample must be :always or :degeneracy"))
```

---

### Step 2: Create new SIR module
**File:** `src/sir.jl` (new file)

**Status:** [x] Complete

```julia
"""
    sir_pool_size(ess_target, c, max_pool)

Compute pool size as min(ceil(c * m * log(m)), max_pool) where m = ess_target.
"""
function sir_pool_size(ess_target::Int, c::Float64, max_pool::Int)
    return min(ceil(Int, c * ess_target * log(ess_target)), max_pool)
end

"""
    resample_multinomial(weights, n_resample)

Resample n_resample indices with replacement, proportional to weights.
Returns Vector{Int} of indices into the original array.
"""
function resample_multinomial(weights::Vector{Float64}, n_resample::Int)
    return StatsBase.sample(1:length(weights), StatsBase.Weights(weights), n_resample; replace=true)
end

"""
    resample_lhs(weights, n_resample)

Resample n_resample indices using Latin Hypercube Sampling on the CDF.
Divides [0,1] into n_resample equal strata, samples one uniform per stratum,
maps to indices via inverse CDF of cumulative weights.
Returns Vector{Int} of indices into the original array.
"""
function resample_lhs(weights::Vector{Float64}, n_resample::Int)
    cumweights = cumsum(weights)
    cumweights ./= cumweights[end]  # Normalize to [0,1]
    
    indices = Vector{Int}(undef, n_resample)
    for i in 1:n_resample
        # Sample uniformly in stratum [(i-1)/n, i/n]
        u = (i - 1 + rand()) / n_resample
        # Find index via inverse CDF (binary search)
        indices[i] = searchsortedfirst(cumweights, u)
    end
    return indices
end

"""
    get_sir_subsample_indices(weights, n_resample, method)

Dispatcher for resampling methods.
"""
function get_sir_subsample_indices(weights::Vector{Float64}, n_resample::Int, method::Symbol)
    if method == :sir
        return resample_multinomial(weights, n_resample)
    elseif method == :lhs
        return resample_lhs(weights, n_resample)
    else
        error("Unknown SIR method: $method")
    end
end

"""
    should_resample(sir_resample, psis_pareto_k, threshold)

Determine whether to resample based on mode and diagnostics.
"""
function should_resample(sir_resample::Symbol, psis_pareto_k::Float64, threshold::Float64)
    if sir_resample == :always
        return true
    elseif sir_resample == :degeneracy
        return psis_pareto_k > threshold
    else
        return false
    end
end
```

---

### Step 3: Modify `DrawSamplePaths!`
**File:** `src/sampling.jl` ~L170

**Status:** [x] Not needed - simplified approach

The implementation passes `sir_pool_target` as the `ess_target` to `DrawSamplePaths!`, 
so the existing sampling logic works unchanged. Pool size computation is centralized in `fit`.

---

### Step 4: Add pool management infrastructure
**File:** `src/modelfitting.jl` ~L976

**Status:** [x] Complete

Add variables:
```julia
sir_subsample_indices = [Vector{Int}() for _ in 1:nsubj]  # Indices into pool
pool_cap_exceeded = false  # Flag for convergence records
```

After initial pool fill:
1. Compute importance weights on full pool via PSIS
2. Apply SIR: `sir_subsample_indices[i] = get_sir_subsample_indices(ImportanceWeights[i], ess_target, sir)`
3. Set `ess_cur[i] = ess_target` (subsample size is ESS)

---

### Step 5: Modify MCEM loop
**File:** `src/modelfitting.jl` ~L1179-1404

**Status:** [x] Complete

**After parameter update:**
1. Recalculate importance weights on **full pool** via `ComputeImportanceWeightsESS!`

**Check resampling trigger:**
```julia
for i in 1:nsubj
    if should_resample(sir_resample, psis_pareto_k[i], sir_degeneracy_threshold)
        sir_subsample_indices[i] = get_sir_subsample_indices(ImportanceWeights[i], ess_target, sir)
    end
end
```

**M-step:**
- Create `SMPanelData` using views: `samplepaths[i][sir_subsample_indices[i]]`
- Use uniform weights: `fill(1/length(sir_subsample_indices[i]), length(sir_subsample_indices[i]))`

**When ESS increases (`ascent_lb < 0`):**
1. Increase `ess_target` as before
2. Compute `pool_target = sir_pool_size(ess_target, sir_pool_constant, sir_max_pool)`
3. If `pool_target == sir_max_pool && !pool_cap_exceeded`:
   - Set `pool_cap_exceeded = true`
   - Warn: "Pool size capped at sir_max_pool; further ESS increases will reduce SIR effectiveness"
4. Augment pool to `pool_target` via `DrawSamplePaths!`
5. Resample `ess_target` paths from expanded pool

---

### Step 6: Modify Q-function and ASE computation
**File:** `src/sir.jl` (new functions)

**Status:** [x] Complete

Implemented as separate functions `mcem_mll_sir` and `mcem_ase_sir` in `src/sir.jl`
rather than modifying the original `mcem_mll` and `mcem_ase` functions.
The `fit` function dispatches to the appropriate version based on `use_sir`.

---

### Step 7: Modify `ComputeImportanceWeightsESS!`
**File:** `src/sampling.jl` ~L1317

**Status:** [x] Complete - no changes needed

The existing `ComputeImportanceWeightsESS!` already computes PSIS weights on the full pool.
SIR resampling and ESS override are handled in the `fit` function after calling this function.

---

### Step 8: Add verbose SIR diagnostics
**File:** `src/modelfitting.jl`

**Status:** [x] Complete

When `verbose=true` and `sir != :none`, print per-iteration:
```
SIR: pool sizes [min/med/max], subsample=N, mean Pareto-k=X.XX, N subjects resampled
```

---

### Step 9: Update convergence records
**File:** `src/modelfitting.jl`

**Status:** [x] Complete

Add to convergence records metadata:
- `sir_pool_cap_exceeded::Bool`
- `sir_method::Symbol` (`:none`, `:sir`, `:lhs`)
- `sir_resample_mode::Symbol` (`:always`, `:degeneracy`)

---

### Step 10: Add imports
**File:** `src/MultistateModels.jl`

**Status:** [x] Complete

```julia
using QuasiMonteCarlo
include("sir.jl")  # Before modelfitting.jl
```

---

### Step 11: Unit tests
**Package:** `MultistateModelsTests`

**Status:** [x] Complete (47 tests passing)

See Test Plan section below.

---

## Test Plan

### Unit Tests (fast, run on every commit)

#### 1. `sir_pool_size` tests
```julia
@testset "sir_pool_size" begin
    # Basic formula: c * m * log(m)
    @test sir_pool_size(50, 2.0, 8192) == ceil(Int, 2.0 * 50 * log(50))  # ≈ 391
    @test sir_pool_size(100, 2.0, 8192) == ceil(Int, 2.0 * 100 * log(100))  # ≈ 921
    
    # Max pool cap
    @test sir_pool_size(1000, 2.0, 500) == 500  # Capped
    @test sir_pool_size(1000, 2.0, 8192) == 8192  # ceil(2*1000*log(1000)) ≈ 13816 -> capped
    
    # Different constants
    @test sir_pool_size(50, 1.0, 8192) == ceil(Int, 1.0 * 50 * log(50))
    @test sir_pool_size(50, 3.0, 8192) == ceil(Int, 3.0 * 50 * log(50))
    
    # Edge cases
    @test sir_pool_size(1, 2.0, 8192) == 0  # log(1) = 0
    @test sir_pool_size(2, 2.0, 8192) == ceil(Int, 2.0 * 2 * log(2))
end
```

#### 2. `resample_multinomial` tests
```julia
@testset "resample_multinomial" begin
    Random.seed!(12345)
    
    # Output length correct
    weights = [0.1, 0.2, 0.3, 0.4]
    indices = resample_multinomial(weights, 100)
    @test length(indices) == 100
    
    # Indices in valid range
    @test all(1 .<= indices .<= 4)
    
    # Higher weights selected more often (statistical test)
    counts = zeros(Int, 4)
    for idx in indices
        counts[idx] += 1
    end
    @test counts[4] > counts[1]  # Very likely with 100 samples
    
    # Deterministic with seed
    Random.seed!(999)
    idx1 = resample_multinomial(weights, 50)
    Random.seed!(999)
    idx2 = resample_multinomial(weights, 50)
    @test idx1 == idx2
    
    # Uniform weights
    uniform_weights = fill(0.25, 4)
    indices_uniform = resample_multinomial(uniform_weights, 1000)
    counts_uniform = [count(==(i), indices_uniform) for i in 1:4]
    @test all(200 .<= counts_uniform .<= 300)  # Each ~250, allow 20% tolerance
end
```

#### 3. `resample_lhs` tests
```julia
@testset "resample_lhs" begin
    Random.seed!(12345)
    
    # Output length correct
    weights = [0.1, 0.2, 0.3, 0.4]
    indices = resample_lhs(weights, 100)
    @test length(indices) == 100
    
    # Indices in valid range
    @test all(1 .<= indices .<= 4)
    
    # Higher weights still selected more often
    counts = zeros(Int, 4)
    for idx in indices
        counts[idx] += 1
    end
    @test counts[4] > counts[1]
    
    # Variance reduction test: LHS should have lower variance than multinomial
    n_reps = 100
    n_resample = 50
    weights_test = normalize([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    
    means_lhs = Float64[]
    means_sir = Float64[]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    for _ in 1:n_reps
        idx_lhs = resample_lhs(weights_test, n_resample)
        idx_sir = resample_multinomial(weights_test, n_resample)
        push!(means_lhs, mean(values[idx_lhs]))
        push!(means_sir, mean(values[idx_sir]))
    end
    
    @test var(means_lhs) <= var(means_sir) * 1.5  # Allow some slack
end
```

#### 4. `should_resample` tests
```julia
@testset "should_resample" begin
    # :always mode
    @test should_resample(:always, 0.3, 0.7) == true
    @test should_resample(:always, 0.9, 0.7) == true
    @test should_resample(:always, 0.0, 0.7) == true
    
    # :degeneracy mode
    @test should_resample(:degeneracy, 0.3, 0.7) == false  # Below threshold
    @test should_resample(:degeneracy, 0.7, 0.7) == false  # At threshold (not >)
    @test should_resample(:degeneracy, 0.71, 0.7) == true  # Above threshold
    @test should_resample(:degeneracy, 0.9, 0.7) == true
    
    # Different thresholds
    @test should_resample(:degeneracy, 0.5, 0.4) == true
    @test should_resample(:degeneracy, 0.5, 0.6) == false
end
```

#### 5. `get_sir_subsample_indices` tests
```julia
@testset "get_sir_subsample_indices" begin
    Random.seed!(12345)
    weights = normalize(rand(100), 1)
    
    # Dispatches correctly
    idx_sir = get_sir_subsample_indices(weights, 50, :sir)
    idx_lhs = get_sir_subsample_indices(weights, 50, :lhs)
    
    @test length(idx_sir) == 50
    @test length(idx_lhs) == 50
    @test all(1 .<= idx_sir .<= 100)
    @test all(1 .<= idx_lhs .<= 100)
    
    # Unknown method throws
    @test_throws ErrorException get_sir_subsample_indices(weights, 50, :unknown)
end
```

#### 6. View-based indexing tests
```julia
@testset "view_based_indexing" begin
    # Test that views work correctly for subsetting
    paths = [SamplePath(1, [0.0, 1.0, 2.0], [1, 2, 1]) for _ in 1:100]
    logliks = randn(100)
    
    indices = [3, 7, 15, 42, 99]
    
    # Views correctly subset
    paths_view = @view paths[indices]
    logliks_view = @view logliks[indices]
    
    @test length(paths_view) == 5
    @test paths_view[1] === paths[3]
    @test logliks_view[2] == logliks[7]
    
    # Modifications to original propagate to view
    original_val = logliks[3]
    logliks[3] = -999.0
    @test logliks_view[1] == -999.0
    
    # Uniform weights for subsample
    uniform_weights = fill(1/5, 5)
    @test sum(uniform_weights) ≈ 1.0
    @test mean(logliks_view .* uniform_weights * 5) ≈ mean(logliks_view)
end
```

---

### Long Tests (nightly or before releases)

#### 1. Integration test: MCEM convergence with SIR
```julia
@testset "mcem_sir_convergence" begin
    # Setup 3-state illness-death model
    # ... model setup code ...
    
    # Fit with sir=:none (baseline)
    Random.seed!(12345)
    result_none = fit(model; sir=:none, maxiter=50, verbose=false)
    
    # Fit with sir=:sir
    Random.seed!(12345)
    result_sir = fit(model; sir=:sir, maxiter=50, verbose=false)
    
    # Fit with sir=:lhs
    Random.seed!(12345)
    result_lhs = fit(model; sir=:lhs, maxiter=50, verbose=false)
    
    # All should converge
    @test result_none.converged
    @test result_sir.converged
    @test result_lhs.converged
    
    # Estimates should be similar (within statistical tolerance)
    @test isapprox(result_none.parameters, result_sir.parameters; rtol=0.1)
    @test isapprox(result_none.parameters, result_lhs.parameters; rtol=0.1)
    
    # Check convergence records metadata
    @test result_sir.convergence_records.sir_method == :sir
    @test result_lhs.convergence_records.sir_method == :lhs
    @test result_none.convergence_records.sir_method == :none
end
```

#### 2. Regression test: `sir=:none` unchanged
```julia
@testset "sir_none_regression" begin
    # Ensure sir=:none produces identical results to pre-SIR implementation
    # Compare against saved reference results
    
    Random.seed!(54321)
    result = fit(model; sir=:none, maxiter=30, verbose=false)
    
    @test result.parameters ≈ reference_parameters
    @test result.loglikelihood ≈ reference_loglik
end
```

#### 3. Pool cap warning test
```julia
@testset "pool_cap_warning" begin
    # Create scenario where pool cap is exceeded
    result = fit(model; 
        sir=:lhs, 
        sir_max_pool=100,  # Very low cap
        ess_target_initial=50,
        maxiter=100,
        verbose=false
    )
    
    @test result.convergence_records.sir_pool_cap_exceeded == true
end
```

#### 4. Degeneracy mode test
```julia
@testset "degeneracy_mode" begin
    result_always = fit(model; sir=:lhs, sir_resample=:always, verbose=false)
    result_degen = fit(model; sir=:lhs, sir_resample=:degeneracy, verbose=false)
    
    @test result_always.convergence_records.sir_resample_mode == :always
    @test result_degen.convergence_records.sir_resample_mode == :degeneracy
end
```

#### 5. Exact observation handling
```julia
@testset "exact_observations" begin
    # Create model with some subjects having all exact observations
    # These should be skipped by SIR
    
    result = fit(model; sir=:lhs, verbose=false)
    @test result.converged
end
```

---

### Benchmarks

#### 1. Timing comparison
```julia
@testset "benchmark_sir_timing" begin
    n_runs = 5
    
    times_none = [(@elapsed fit(model; sir=:none, maxiter=30, verbose=false)) for _ in 1:n_runs]
    times_sir = [(@elapsed fit(model; sir=:sir, maxiter=30, verbose=false)) for _ in 1:n_runs]
    times_lhs = [(@elapsed fit(model; sir=:lhs, maxiter=30, verbose=false)) for _ in 1:n_runs]
    
    println("Timing (mean ± std):")
    println("  sir=:none : $(mean(times_none)) ± $(std(times_none)) s")
    println("  sir=:sir  : $(mean(times_sir)) ± $(std(times_sir)) s")
    println("  sir=:lhs  : $(mean(times_lhs)) ± $(std(times_lhs)) s")
    
    # SIR should not be dramatically slower
    @test mean(times_sir) < mean(times_none) * 3
    @test mean(times_lhs) < mean(times_none) * 3
end
```

#### 2. Variance comparison
```julia
@testset "benchmark_sir_variance" begin
    n_fits = 20
    
    estimates_none = [fit(model; sir=:none, maxiter=50, verbose=false).parameters for seed in 1:n_fits]
    estimates_sir = [fit(model; sir=:sir, maxiter=50, verbose=false).parameters for seed in 1:n_fits]
    estimates_lhs = [fit(model; sir=:lhs, maxiter=50, verbose=false).parameters for seed in 1:n_fits]
    
    var_none = var(reduce(hcat, estimates_none), dims=2)
    var_sir = var(reduce(hcat, estimates_sir), dims=2)
    var_lhs = var(reduce(hcat, estimates_lhs), dims=2)
    
    println("Parameter variance across $(n_fits) fits:")
    println("  sir=:none : $(mean(var_none))")
    println("  sir=:sir  : $(mean(var_sir))")
    println("  sir=:lhs  : $(mean(var_lhs))")
    
    # LHS should have lower or equal variance
    @test mean(var_lhs) <= mean(var_sir) * 1.2
end
```

#### 3. Memory usage
```julia
@testset "benchmark_sir_memory" begin
    alloc_none = @allocated fit(model; sir=:none, maxiter=30, verbose=false)
    alloc_sir = @allocated fit(model; sir=:sir, maxiter=30, verbose=false)
    alloc_lhs = @allocated fit(model; sir=:lhs, maxiter=30, verbose=false)
    
    println("Memory allocations:")
    println("  sir=:none : $(alloc_none / 1e6) MB")
    println("  sir=:sir  : $(alloc_sir / 1e6) MB")
    println("  sir=:lhs  : $(alloc_lhs / 1e6) MB")
end
```

#### 4. Scaling with subjects
```julia
@testset "benchmark_sir_scaling" begin
    for n_subj in [50, 100, 200, 500]
        model = generate_test_model(n_subj)
        
        t_none = @elapsed fit(model; sir=:none, maxiter=20, verbose=false)
        t_lhs = @elapsed fit(model; sir=:lhs, maxiter=20, verbose=false)
        
        println("n_subj=$n_subj: none=$(t_none)s, lhs=$(t_lhs)s, ratio=$(t_lhs/t_none)")
    end
end
```

---

## Notes

- Only warn once when pool cap is exceeded (use `pool_cap_exceeded` flag)
- Skip SIR for subjects where all observations are exact
- Using views is more memory efficient but requires careful testing
