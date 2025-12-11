# Profile Likelihood Computation Performance
# Run with: julia --project=. scratch/profiling/profile_likelihood.jl

using MultistateModels
using MultistateModels: loglik, loglik_exact, loglik_AD, MPanelData, SMPanelData,
                        build_tpm_mapping, safe_unflatten, make_subjdat,
                        ExactData, ExactDataAD, get_parameters_flat,
                        eval_hazard, eval_cumhaz
using Profile
using BenchmarkTools
using Printf
using JSON
using Dates

# Include fixtures
include("setup_profiling_fixtures.jl")
using .ProfilingFixtures

println("="^70)
println("LIKELIHOOD PROFILING - Phase 0 Baseline")
println("="^70)
println("Date: ", Dates.now())
println()

# Store results
results = Dict{String, Any}()

# -----------------------------------------------------------------------------
# Benchmark: Markov panel likelihood
# -----------------------------------------------------------------------------
println("\n--- Markov Panel Likelihood ---\n")

markov_results = Dict{String, Any}()

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
    loglik(params, data)
    
    # Benchmark
    b = @benchmark loglik($params, $data)
    
    median_us = median(b.times)/1000
    allocs = b.allocs
    mem_kb = b.memory/1024
    
    @printf("%-30s: %8.2f μs, %6d allocs, %8.2f KiB\n",
            name, median_us, allocs, mem_kb)
    
    markov_results[name] = Dict(
        "median_us" => median_us,
        "allocs" => allocs,
        "memory_kb" => mem_kb
    )
end

results["markov_likelihood"] = markov_results

# -----------------------------------------------------------------------------
# Benchmark: Semi-Markov likelihood (with sampled paths)
# -----------------------------------------------------------------------------
println("\n--- Semi-Markov Likelihood (sampled paths) ---\n")

semimarkov_results = Dict{String, Any}()

for (name, create_fn, nsubj) in [
    ("2-state Semi-Markov (n=50)", () -> create_semimarkov_2state(nsubj=50), 50),
    ("3-state Semi-Markov + Cov (n=50)", () -> create_semimarkov_with_covariates(nsubj=50), 50),
]
    model = create_fn()
    set_surrogate!(model)
    
    # Simulate paths - returns Vector{Vector{SamplePath}}
    # where outer is simulation replicates, inner is subjects
    trajectories = simulate(model; nsim=1, paths=true, data=false)
    paths = trajectories[1]  # Get the first replicate
    
    # SMPanelData expects Vector{Vector{SamplePath}} where
    # outer is subjects, inner is multiple sample paths per subject
    samplepaths = [[p] for p in paths]
    ImportanceWeights = [[1.0] for _ in paths]
    
    sm_data = SMPanelData(model, samplepaths, ImportanceWeights)
    params = get_parameters_flat(model)
    
    # Warm-up
    loglik(params, sm_data)
    
    # Benchmark
    b = @benchmark loglik($params, $sm_data)
    
    median_us = median(b.times)/1000
    allocs = b.allocs
    mem_kb = b.memory/1024
    
    @printf("%-40s: %8.2f μs, %6d allocs, %8.2f KiB\n",
            name, median_us, allocs, mem_kb)
    
    semimarkov_results[name] = Dict(
        "median_us" => median_us,
        "allocs" => allocs,
        "memory_kb" => mem_kb
    )
end

results["semimarkov_likelihood"] = semimarkov_results

# -----------------------------------------------------------------------------
# Benchmark: Exact data likelihood
# -----------------------------------------------------------------------------
println("\n--- Exact Data Likelihood ---\n")

exact_results = Dict{String, Any}()

# Create a model with exact data
model = create_semimarkov_2state(nsubj=50)
set_surrogate!(model)

# Generate exact paths
trajectories = simulate(model; nsim=1, paths=true, data=false)
paths = trajectories[1]  # Get first replicate (Vector{SamplePath})

# Create ExactData structure  
exact_data = ExactData(model, paths)
params = get_parameters_flat(model)

# Warm-up
loglik_exact(params, exact_data)

# Benchmark
b = @benchmark loglik_exact($params, $exact_data)

median_us = median(b.times)/1000
allocs = b.allocs
mem_kb = b.memory/1024

@printf("ExactData (n=50):              %8.2f μs, %6d allocs, %8.2f KiB\n",
        median_us, allocs, mem_kb)

exact_results["ExactData (n=50)"] = Dict(
    "median_us" => median_us,
    "allocs" => allocs,
    "memory_kb" => mem_kb
)

# Also test ExactDataAD (functional version for single path)
# ExactDataAD is designed for single paths for variance estimation
single_path = [paths[1]]
exact_data_ad = ExactDataAD(single_path, [1.0], model.hazards, model)
loglik_AD(params, exact_data_ad)

b = @benchmark loglik_AD($params, $exact_data_ad)

median_us = median(b.times)/1000
allocs = b.allocs
mem_kb = b.memory/1024

@printf("ExactDataAD (single path):     %8.2f μs, %6d allocs, %8.2f KiB\n",
        median_us, allocs, mem_kb)

exact_results["ExactDataAD (single_path)"] = Dict(
    "median_us" => median_us,
    "allocs" => allocs,
    "memory_kb" => mem_kb
)

results["exact_likelihood"] = exact_results

# Skip hazard evaluation benchmark - requires proper parameter structure setup
# Will profile hazard evaluation as part of full likelihood profiling

# -----------------------------------------------------------------------------
# Profile: Likelihood hot path
# -----------------------------------------------------------------------------
println("\n--- Profiling Likelihood Hot Path ---\n")

model = create_markov_3state(nsubj=200)
books = build_tpm_mapping(model.data)
data = MPanelData(model, books)
params = get_parameters_flat(model)

Profile.clear()
@profile for _ in 1:500
    loglik(params, data)
end

println("Top 20 functions by time:")
Profile.print(maxdepth=20, mincount=50, noisefloor=2.0)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
results["metadata"] = Dict(
    "date" => string(Dates.now()),
    "julia_version" => string(VERSION),
    "phase" => "0_baseline"
)

open("scratch/profiling/likelihood_baseline.json", "w") do f
    JSON.print(f, results, 2)
end

println("\n" * "="^70)
println("LIKELIHOOD PROFILING COMPLETE")
println("Results saved to scratch/profiling/likelihood_baseline.json")
println("="^70)
