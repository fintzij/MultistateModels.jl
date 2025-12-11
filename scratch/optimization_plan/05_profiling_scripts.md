# Part 5: Profiling Scripts

## 5.1 Setup Script

Create test fixtures for profiling:

```julia
# scratch/profiling/setup_profiling_fixtures.jl

using MultistateModels
using DataFrames
using Random

"""
Create fixtures for profiling different model configurations.
"""
module ProfilingFixtures

using MultistateModels
using DataFrames
using Random

export create_markov_2state, create_markov_3state, 
       create_semimarkov_2state, create_semimarkov_3state,
       create_semimarkov_with_covariates, create_spline_model

# -----------------------------------------------------------------------------
# 2-State Markov Model (Exponential)
# -----------------------------------------------------------------------------
function create_markov_2state(; nsubj=100, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), :exp, 1, 2)
    
    # Generate data
    data = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 5.0], nsubj),
        tstop = repeat([5.0, 10.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 2], nsubj)
    )
    
    # Randomize outcomes
    for i in 1:nsubj
        if rand() < 0.3
            data[(i-1)*2 + 1, :stateto] = 2
            data[(i-1)*2 + 2, :statefrom] = 2
            data[(i-1)*2 + 2, :stateto] = 2
        end
    end
    
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [log(0.1)],))
    
    return model
end

# -----------------------------------------------------------------------------
# 3-State Progressive Markov Model
# -----------------------------------------------------------------------------
function create_markov_3state(; nsubj=100, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), :exp, 1, 2)
    h23 = Hazard(@formula(0 ~ 1), :exp, 2, 3)
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=3),
        tstart = repeat([0.0, 3.0, 6.0], nsubj),
        tstop = repeat([3.0, 6.0, 10.0], nsubj),
        statefrom = repeat([1, 1, 1], nsubj),
        stateto = repeat([1, 1, 1], nsubj),
        obstype = repeat([2, 2, 2], nsubj)
    )
    
    # Randomize progressions
    for i in 1:nsubj
        base = (i-1)*3
        if rand() < 0.4
            data[base + 1, :stateto] = 2
            data[base + 2, :statefrom] = 2
            if rand() < 0.5
                data[base + 2, :stateto] = 3
                data[base + 3, :statefrom] = 3
                data[base + 3, :stateto] = 3
            else
                data[base + 2, :stateto] = 2
                data[base + 3, :statefrom] = 2
                data[base + 3, :stateto] = rand() < 0.5 ? 3 : 2
            end
        end
    end
    
    model = multistatemodel(h12, h23; data=data)
    set_parameters!(model, (h12 = [log(0.15)], h23 = [log(0.2)]))
    
    return model
end

# -----------------------------------------------------------------------------
# 2-State Semi-Markov Model (Weibull)
# -----------------------------------------------------------------------------
function create_semimarkov_2state(; nsubj=100, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), :wei, 1, 2)
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 5.0], nsubj),
        tstop = repeat([5.0, 10.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 2], nsubj)
    )
    
    for i in 1:nsubj
        if rand() < 0.3
            data[(i-1)*2 + 1, :stateto] = 2
            data[(i-1)*2 + 2, :statefrom] = 2
            data[(i-1)*2 + 2, :stateto] = 2
        end
    end
    
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [log(1.5), log(0.1)],))  # shape=1.5, scale=0.1
    set_surrogate!(model)
    
    return model
end

# -----------------------------------------------------------------------------
# 3-State Semi-Markov with Covariates
# -----------------------------------------------------------------------------
function create_semimarkov_with_covariates(; nsubj=200, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1 + age + trt), :wei, 1, 2)
    h23 = Hazard(@formula(0 ~ 1 + age), :gom, 2, 3)
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=3),
        tstart = repeat([0.0, 3.0, 6.0], nsubj),
        tstop = repeat([3.0, 6.0, 10.0], nsubj),
        statefrom = repeat([1, 1, 1], nsubj),
        stateto = repeat([1, 1, 1], nsubj),
        obstype = repeat([2, 2, 2], nsubj),
        age = repeat(randn(nsubj), inner=3),
        trt = repeat(rand([0, 1], nsubj), inner=3)
    )
    
    # Randomize outcomes
    for i in 1:nsubj
        base = (i-1)*3
        if rand() < 0.4
            data[base + 1, :stateto] = 2
            data[base + 2, :statefrom] = 2
            if rand() < 0.5
                data[base + 2, :stateto] = 3
                data[base + 3, :statefrom] = 3
                data[base + 3, :stateto] = 3
            else
                data[base + 2, :stateto] = 2
                data[base + 3, :statefrom] = 2
                data[base + 3, :stateto] = rand() < 0.5 ? 3 : 2
            end
        end
    end
    
    model = multistatemodel(h12, h23; data=data)
    set_parameters!(model, (
        h12 = [log(1.2), log(0.15), 0.1, -0.3],  # shape, scale, age, trt
        h23 = [log(0.05), log(0.1), 0.2]          # shape, rate, age
    ))
    set_surrogate!(model)
    
    return model
end

# -----------------------------------------------------------------------------
# Spline Model
# -----------------------------------------------------------------------------
function create_spline_model(; nsubj=100, nknots=3, seed=12345)
    Random.seed!(seed)
    
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2; 
                 df=nknots+1, degree=3, extrapolation="constant")
    
    data = DataFrame(
        id = repeat(1:nsubj, inner=2),
        tstart = repeat([0.0, 5.0], nsubj),
        tstop = repeat([5.0, 10.0], nsubj),
        statefrom = repeat([1, 1], nsubj),
        stateto = repeat([1, 2], nsubj),
        obstype = repeat([2, 2], nsubj)
    )
    
    for i in 1:nsubj
        if rand() < 0.3
            data[(i-1)*2 + 1, :stateto] = 2
            data[(i-1)*2 + 2, :statefrom] = 2
            data[(i-1)*2 + 2, :stateto] = 2
        end
    end
    
    model = multistatemodel(h12; data=data)
    set_surrogate!(model)
    
    return model
end

end # module
```

---

## 5.2 Simulation Profiling Script

```julia
# scratch/profiling/profile_simulation.jl
#
# Profile simulation performance
# Run with: julia --project=. scratch/profiling/profile_simulation.jl

using MultistateModels
using Profile
using BenchmarkTools
using Printf

# Include fixtures
include("setup_profiling_fixtures.jl")
using .ProfilingFixtures

println("="^60)
println("SIMULATION PROFILING")
println("="^60)

# -----------------------------------------------------------------------------
# Benchmark: Single path simulation
# -----------------------------------------------------------------------------
println("\n--- Single Path Simulation ---\n")

for (name, create_fn) in [
    ("2-state Markov", () -> create_markov_2state(nsubj=100)),
    ("3-state Markov", () -> create_markov_3state(nsubj=100)),
    ("2-state Semi-Markov (Weibull)", () -> create_semimarkov_2state(nsubj=100)),
    ("3-state Semi-Markov + Covariates", () -> create_semimarkov_with_covariates(nsubj=100)),
]
    model = create_fn()
    
    # Warm-up
    simulate_path(model, 1)
    
    # Benchmark
    b = @benchmark simulate_path($model, 1)
    @printf("%-35s: %8.2f μs (median), %6d allocs, %8.2f KiB\n",
            name, median(b.times)/1000, b.allocs, b.memory/1024)
end

# -----------------------------------------------------------------------------
# Benchmark: Multi-path simulation
# -----------------------------------------------------------------------------
println("\n--- Multi-Path Simulation (nsim=100) ---\n")

for (name, create_fn) in [
    ("2-state Markov", () -> create_markov_2state(nsubj=50)),
    ("3-state Markov", () -> create_markov_3state(nsubj=50)),
    ("2-state Semi-Markov", () -> create_semimarkov_2state(nsubj=50)),
]
    model = create_fn()
    
    # Warm-up
    simulate(model; nsim=1)
    
    # Benchmark
    b = @benchmark simulate($model; nsim=100)
    @printf("%-35s: %8.2f ms (median), %8d allocs, %8.2f MiB\n",
            name, median(b.times)/1e6, b.allocs, b.memory/1024/1024)
end

# -----------------------------------------------------------------------------
# Profile: Detailed simulation breakdown
# -----------------------------------------------------------------------------
println("\n--- Profiling Simulation Hot Path ---\n")

model = create_semimarkov_with_covariates(nsubj=100)

# Clear and collect profile
Profile.clear()
@profile for _ in 1:50
    simulate(model; nsim=10)
end

# Print profile summary
println("Top 20 functions by time:")
Profile.print(maxdepth=20, mincount=100, noisefloor=2.0)

# If ProfileView is available, show flamegraph
try
    using ProfileView
    println("\nOpening flamegraph...")
    ProfileView.view()
catch
    println("\nInstall ProfileView for interactive flamegraph: ] add ProfileView")
end

# -----------------------------------------------------------------------------
# Allocation tracking
# -----------------------------------------------------------------------------
println("\n--- Allocation Tracking ---\n")

model = create_semimarkov_2state(nsubj=50)

# Track allocations with @time
println("Single simulate_path:")
@time simulate_path(model, 1)
@time simulate_path(model, 1)

println("\nSimulate nsim=100:")
@time simulate(model; nsim=100)

println("\n" * "="^60)
println("SIMULATION PROFILING COMPLETE")
println("="^60)
```

---

## 5.3 Likelihood Profiling Script

```julia
# scratch/profiling/profile_likelihood.jl
#
# Profile likelihood computation performance
# Run with: julia --project=. scratch/profiling/profile_likelihood.jl

using MultistateModels
using MultistateModels: loglik_markov, loglik_markov_functional, 
                        loglik_path, MPanelData, build_tpm_mapping,
                        safe_unflatten, make_subjdat
using Profile
using BenchmarkTools
using Printf

include("setup_profiling_fixtures.jl")
using .ProfilingFixtures

println("="^60)
println("LIKELIHOOD PROFILING")
println("="^60)

# -----------------------------------------------------------------------------
# Benchmark: Markov panel likelihood
# -----------------------------------------------------------------------------
println("\n--- Markov Panel Likelihood ---\n")

for (name, create_fn) in [
    ("2-state Markov (n=100)", () -> create_markov_2state(nsubj=100)),
    ("3-state Markov (n=100)", () -> create_markov_3state(nsubj=100)),
    ("2-state Markov (n=500)", () -> create_markov_2state(nsubj=500)),
]
    model = create_fn()
    books = build_tpm_mapping(model.data)
    data = MPanelData(model, books)
    params = get_parameters_flat(model)
    
    # Warm-up
    loglik_markov(params, data)
    
    # Benchmark mutating version
    b_mut = @benchmark loglik_markov($params, $data)
    
    # Benchmark functional version (if exists)
    b_func = @benchmark loglik_markov_functional($params, $data)
    
    @printf("%-30s\n", name)
    @printf("  Mutating:   %8.2f μs, %6d allocs, %8.2f KiB\n",
            median(b_mut.times)/1000, b_mut.allocs, b_mut.memory/1024)
    @printf("  Functional: %8.2f μs, %6d allocs, %8.2f KiB\n",
            median(b_func.times)/1000, b_func.allocs, b_func.memory/1024)
    @printf("  Ratio: %.2fx\n\n", median(b_func.times)/median(b_mut.times))
end

# -----------------------------------------------------------------------------
# Benchmark: Path likelihood (exact data)
# -----------------------------------------------------------------------------
println("\n--- Path Likelihood ---\n")

model = create_semimarkov_with_covariates(nsubj=100)
paths = simulate(model; nsim=1, data=false, paths=true)[1]
path = paths[1]
params = safe_unflatten(get_parameters_flat(model), model)

# Get subject data
subj_inds = model.subjectindices[path.subj]
subj_dat = view(model.data, subj_inds, :)
subjdat_df = make_subjdat(path, subj_dat)

# Benchmark loglik_path
b = @benchmark loglik_path($params, $subjdat_df, $model.hazards, 
                           $model.totalhazards, $model.tmat)
@printf("loglik_path: %8.2f μs, %6d allocs, %8.2f KiB\n",
        median(b.times)/1000, b.allocs, b.memory/1024)

# Benchmark make_subjdat
b = @benchmark make_subjdat($path, $subj_dat)
@printf("make_subjdat: %8.2f μs, %6d allocs, %8.2f KiB\n",
        median(b.times)/1000, b.allocs, b.memory/1024)

# -----------------------------------------------------------------------------
# Benchmark: Hazard evaluation
# -----------------------------------------------------------------------------
println("\n--- Hazard Evaluation ---\n")

using MultistateModels: eval_hazard, eval_cumhaz, extract_covariates_fast

# Exponential hazard
model_exp = create_markov_2state(nsubj=10)
hazard_exp = model_exp.hazards[1]
params_exp = safe_unflatten(get_parameters_flat(model_exp), model_exp)
pars_exp = params_exp[hazard_exp.hazname]
row = model_exp.data[1, :]

b = @benchmark eval_hazard($hazard_exp, 1.0, $pars_exp, $row)
@printf("eval_hazard (exp): %8.2f ns, %d allocs\n", median(b.times), b.allocs)

b = @benchmark eval_cumhaz($hazard_exp, 0.0, 1.0, $pars_exp, $row)
@printf("eval_cumhaz (exp): %8.2f ns, %d allocs\n", median(b.times), b.allocs)

# Weibull hazard
model_wei = create_semimarkov_2state(nsubj=10)
hazard_wei = model_wei.hazards[1]
params_wei = safe_unflatten(get_parameters_flat(model_wei), model_wei)
pars_wei = params_wei[hazard_wei.hazname]
row_wei = model_wei.data[1, :]

b = @benchmark eval_hazard($hazard_wei, 1.0, $pars_wei, $row_wei)
@printf("eval_hazard (wei): %8.2f ns, %d allocs\n", median(b.times), b.allocs)

b = @benchmark eval_cumhaz($hazard_wei, 0.0, 1.0, $pars_wei, $row_wei)
@printf("eval_cumhaz (wei): %8.2f ns, %d allocs\n", median(b.times), b.allocs)

# Covariate extraction
model_cov = create_semimarkov_with_covariates(nsubj=10)
hazard_cov = model_cov.hazards[1]
row_cov = model_cov.data[1, :]

b = @benchmark extract_covariates_fast($row_cov, $(hazard_cov.covar_names))
@printf("extract_covariates_fast: %8.2f ns, %d allocs\n", median(b.times), b.allocs)

# -----------------------------------------------------------------------------
# Profile: Likelihood hot path
# -----------------------------------------------------------------------------
println("\n--- Profiling Likelihood Hot Path ---\n")

model = create_markov_3state(nsubj=200)
books = build_tpm_mapping(model.data)
data = MPanelData(model, books)
params = get_parameters_flat(model)

Profile.clear()
@profile for _ in 1:1000
    loglik_markov(params, data)
end

println("Top 20 functions by time:")
Profile.print(maxdepth=20, mincount=100, noisefloor=2.0)

println("\n" * "="^60)
println("LIKELIHOOD PROFILING COMPLETE")
println("="^60)
```

---

## 5.4 MCEM Profiling Script

```julia
# scratch/profiling/profile_mcem.jl
#
# Profile MCEM inference performance
# Run with: julia --project=. scratch/profiling/profile_mcem.jl

using MultistateModels
using MultistateModels: DrawSamplePaths!, loglik_semi_markov, SMPanelData,
                        build_tpm_mapping, build_hazmat_book, build_tpm_book,
                        compute_hazmat!, compute_tmat!, safe_unflatten,
                        ComputeImportanceWeightsESS!
using Profile
using BenchmarkTools
using Printf

include("setup_profiling_fixtures.jl")
using .ProfilingFixtures

println("="^60)
println("MCEM PROFILING")
println("="^60)

# -----------------------------------------------------------------------------
# Setup MCEM infrastructure
# -----------------------------------------------------------------------------
println("\n--- Setting up MCEM infrastructure ---\n")

model = create_semimarkov_with_covariates(nsubj=50)

# Build TPM infrastructure
books = build_tpm_mapping(model.data)
hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
tpm_book = build_tpm_book(Float64, model.tmat, books[1])

# Compute surrogate TPMs
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
surrogate = model.markovsurrogate
surrogate_pars = safe_unflatten(get_parameters_flat(surrogate), surrogate)

for t in eachindex(books[1])
    compute_hazmat!(hazmat_book[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
    compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
end

println("Infrastructure ready.")

# -----------------------------------------------------------------------------
# Benchmark: TPM computation
# -----------------------------------------------------------------------------
println("\n--- TPM Computation ---\n")

b = @benchmark begin
    for t in eachindex($books[1])
        compute_hazmat!($hazmat_book[t], $surrogate_pars, $(surrogate.hazards), 
                        $books[1][t], $(model.data))
        compute_tmat!($tpm_book[t], $hazmat_book[t], $books[1][t], $cache)
    end
end
@printf("All TPMs: %8.2f μs, %6d allocs, %8.2f KiB\n",
        median(b.times)/1000, b.allocs, b.memory/1024)

# Single TPM
b = @benchmark compute_hazmat!($(hazmat_book[1]), $surrogate_pars, 
                               $(surrogate.hazards), $(books[1][1]), $(model.data))
@printf("Single hazmat: %8.2f ns, %d allocs\n", median(b.times), b.allocs)

b = @benchmark compute_tmat!($(tpm_book[1]), $(hazmat_book[1]), $(books[1][1]), $cache)
@printf("Single tmat (exp): %8.2f ns, %d allocs\n", median(b.times), b.allocs)

# -----------------------------------------------------------------------------
# Benchmark: E-step (path sampling)
# -----------------------------------------------------------------------------
println("\n--- E-Step: Path Sampling ---\n")

using MultistateModels: draw_samplepath, build_fbmats, sample_ecctmc!

# Build FFBS infrastructure
nsubj = length(model.subjectindices)
absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

if any(model.data.obstype .> 2)
    fbmats = build_fbmats(model)
else
    fbmats = nothing
end

# Benchmark single path draw
b = @benchmark draw_samplepath(1, $model, $tpm_book, $hazmat_book, 
                               $(books[2]), $fbmats, $absorbingstates)
@printf("draw_samplepath: %8.2f μs, %6d allocs, %8.2f KiB\n",
        median(b.times)/1000, b.allocs, b.memory/1024)

# Benchmark sample_ecctmc (uniformization)
P = tpm_book[1][1]
Q = hazmat_book[1]
a, b_state = 1, 2
t0, t1 = 0.0, 1.0

times_vec = Float64[]
states_vec = Int[]
bench = @benchmark begin
    empty!($times_vec)
    empty!($states_vec)
    sample_ecctmc!($times_vec, $states_vec, $P, $Q, $a, $b_state, $t0, $t1)
end
@printf("sample_ecctmc!: %8.2f μs, %6d allocs, %8.2f KiB\n",
        median(bench.times)/1000, bench.allocs, bench.memory/1024)

# -----------------------------------------------------------------------------
# Benchmark: M-step (likelihood)
# -----------------------------------------------------------------------------
println("\n--- M-Step: Likelihood Evaluation ---\n")

# Draw some paths first
paths = simulate(model; nsim=1, data=false, paths=true)[1]
samplepaths = [[p] for p in paths]
ImportanceWeights = [[1.0] for _ in paths]

sm_data = SMPanelData(model, samplepaths, ImportanceWeights)
params = get_parameters_flat(model)

b = @benchmark loglik_semi_markov($params, $sm_data)
@printf("loglik_semi_markov: %8.2f ms, %8d allocs, %8.2f MiB\n",
        median(b.times)/1e6, b.allocs, b.memory/1024/1024)

# Per-path breakdown
path = paths[1]
params_nested = safe_unflatten(params, model)
subj_inds = model.subjectindices[path.subj]
subj_dat = view(model.data, subj_inds, :)

using MultistateModels: make_subjdat, loglik_path

b = @benchmark make_subjdat($path, $subj_dat)
@printf("  make_subjdat: %8.2f μs, %d allocs\n", median(b.times)/1000, b.allocs)

subjdat_df = make_subjdat(path, subj_dat)
b = @benchmark loglik_path($params_nested, $subjdat_df, $(model.hazards), 
                           $(model.totalhazards), $(model.tmat))
@printf("  loglik_path: %8.2f μs, %d allocs\n", median(b.times)/1000, b.allocs)

# -----------------------------------------------------------------------------
# Profile: Full MCEM iteration
# -----------------------------------------------------------------------------
println("\n--- Profiling Full MCEM Iteration ---\n")

# Run a few MCEM iterations with profiling
# Note: This is expensive, limit iterations
model_profile = create_semimarkov_2state(nsubj=30)

Profile.clear()
@profile fit(model_profile; maxiter=3, tol=1e-1, verbose=false)

println("Top 30 functions by time:")
Profile.print(maxdepth=30, mincount=50, noisefloor=2.0)

println("\n" * "="^60)
println("MCEM PROFILING COMPLETE")
println("="^60)
```

---

## 5.5 Running the Profiles

### Quick Profile (5 minutes)

```bash
cd "/Users/fintzij/Library/CloudStorage/OneDrive-BristolMyersSquibb/Documents/Julia packages/MultistateModels.jl"

# Create profiling directory
mkdir -p scratch/profiling

# Run simulation profile
julia --project=. scratch/profiling/profile_simulation.jl | tee scratch/profiling/simulation_results.txt

# Run likelihood profile
julia --project=. scratch/profiling/profile_likelihood.jl | tee scratch/profiling/likelihood_results.txt
```

### Full Profile (30 minutes)

```bash
# Run MCEM profile (slower)
julia --project=. scratch/profiling/profile_mcem.jl | tee scratch/profiling/mcem_results.txt
```

### Interactive Profile with Flamegraph

```julia
# In Julia REPL
using ProfileView
include("scratch/profiling/profile_simulation.jl")
# ProfileView.view() opens automatically
```

---

## 5.6 Interpreting Results

### Key Metrics to Watch

| Metric | Healthy | Concerning |
|--------|---------|------------|
| Allocations per path | < 100 | > 500 |
| Time per simulate_path | < 1 ms | > 10 ms |
| Allocations per loglik_path | < 50 | > 200 |
| loglik_markov_functional / loglik_markov ratio | < 1.2 | > 2.0 |
| make_subjdat time | < 10 μs | > 100 μs |

### Common Issues to Look For

1. **High allocations in inner loops**: Check for temporary array creation
2. **Dict lookup overhead**: `pars[hazard.hazname]` in tight loops
3. **DataFrame row access**: Should use column vectors or views
4. **Problem construction overhead**: `IntervalNonlinearProblem` in `_find_jump_time`
5. **Type instability**: Look for `Any` or `Union` in profile output
