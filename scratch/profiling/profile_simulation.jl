# Profile Simulation Performance
# Run with: julia --project=. scratch/profiling/profile_simulation.jl

using MultistateModels
using Profile
using BenchmarkTools
using Printf
using JSON
using Dates

# Include fixtures
include("setup_profiling_fixtures.jl")
using .ProfilingFixtures

println("="^70)
println("SIMULATION PROFILING - Phase 0 Baseline")
println("="^70)
println("Date: ", Dates.now())
println()

# Store results
results = Dict{String, Any}()

# -----------------------------------------------------------------------------
# Benchmark: Single path simulation
# -----------------------------------------------------------------------------
println("\n--- Single Path Simulation ---\n")

single_path_results = Dict{String, Any}()

for (name, create_fn, needs_surrogate) in [
    ("2-state Markov", () -> create_markov_2state(nsubj=100), false),
    ("3-state Markov", () -> create_markov_3state(nsubj=100), false),
    ("2-state Semi-Markov (Weibull)", () -> create_semimarkov_2state(nsubj=100), true),
    ("3-state Semi-Markov + Covariates", () -> create_semimarkov_with_covariates(nsubj=100), true),
]
    model = create_fn()
    
    # Fit surrogate if semi-Markov
    if needs_surrogate
        set_surrogate!(model)
    end
    
    # Warm-up
    simulate(model; nsim=1, paths=true, data=false)
    
    # Benchmark
    b = @benchmark simulate($model; nsim=1, paths=true, data=false)
    
    median_us = median(b.times)/1000
    allocs = b.allocs
    mem_kb = b.memory/1024
    
    @printf("%-35s: %8.2f μs (median), %6d allocs, %8.2f KiB\n",
            name, median_us, allocs, mem_kb)
    
    single_path_results[name] = Dict(
        "median_us" => median_us,
        "allocs" => allocs,
        "memory_kb" => mem_kb
    )
end

results["single_path"] = single_path_results

# -----------------------------------------------------------------------------
# Benchmark: Multi-path simulation
# -----------------------------------------------------------------------------
println("\n--- Multi-Path Simulation (nsim=100) ---\n")

multi_path_results = Dict{String, Any}()

for (name, create_fn, needs_surrogate) in [
    ("2-state Markov (n=50)", () -> create_markov_2state(nsubj=50), false),
    ("3-state Markov (n=50)", () -> create_markov_3state(nsubj=50), false),
    ("2-state Semi-Markov (n=50)", () -> create_semimarkov_2state(nsubj=50), true),
]
    model = create_fn()
    
    if needs_surrogate
        set_surrogate!(model)
    end
    
    # Warm-up
    simulate(model; nsim=1, paths=true, data=false)
    
    # Benchmark
    b = @benchmark simulate($model; nsim=100, paths=true, data=false)
    
    median_ms = median(b.times)/1e6
    allocs = b.allocs
    mem_mb = b.memory/1024/1024
    
    @printf("%-35s: %8.2f ms (median), %8d allocs, %8.2f MiB\n",
            name, median_ms, allocs, mem_mb)
    
    multi_path_results[name] = Dict(
        "median_ms" => median_ms,
        "allocs" => allocs,
        "memory_mb" => mem_mb
    )
end

results["multi_path_100"] = multi_path_results

# -----------------------------------------------------------------------------
# Benchmark: Scaling with nsim
# -----------------------------------------------------------------------------
println("\n--- Scaling with nsim (2-state Semi-Markov, n=50) ---\n")

scaling_results = Dict{String, Any}()
model = create_semimarkov_2state(nsubj=50)
set_surrogate!(model)

for nsim in [10, 50, 100, 200]
    simulate(model; nsim=1, paths=true, data=false)  # warm-up
    
    b = @benchmark simulate($model; nsim=$nsim, paths=true, data=false)
    
    median_ms = median(b.times)/1e6
    allocs = b.allocs
    mem_mb = b.memory/1024/1024
    
    @printf("nsim=%3d: %8.2f ms, %8d allocs, %8.2f MiB\n",
            nsim, median_ms, allocs, mem_mb)
    
    scaling_results["nsim_$nsim"] = Dict(
        "median_ms" => median_ms,
        "allocs" => allocs,
        "memory_mb" => mem_mb
    )
end

results["scaling_nsim"] = scaling_results

# -----------------------------------------------------------------------------
# Profile: Detailed simulation breakdown
# -----------------------------------------------------------------------------
println("\n--- Profiling Simulation Hot Path ---\n")

model = create_semimarkov_with_covariates(nsubj=100)
set_surrogate!(model)

# Clear and collect profile
Profile.clear()
@profile for _ in 1:20
    simulate(model; nsim=10, paths=true, data=false)
end

# Print profile summary
println("Top 20 functions by time (showing flat view):")
Profile.print(maxdepth=20, mincount=50, noisefloor=2.0)

# -----------------------------------------------------------------------------
# Allocation tracking
# -----------------------------------------------------------------------------
println("\n--- Allocation Tracking ---\n")

model = create_semimarkov_2state(nsubj=50)
set_surrogate!(model)

println("Single simulate (nsim=1):")
@time simulate(model; nsim=1, paths=true, data=false)
@time simulate(model; nsim=1, paths=true, data=false)

println("\nSimulate nsim=100:")
@time simulate(model; nsim=100, paths=true, data=false)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
results["metadata"] = Dict(
    "date" => string(Dates.now()),
    "julia_version" => string(VERSION),
    "phase" => "0_baseline"
)

open("scratch/profiling/simulation_baseline.json", "w") do f
    JSON.print(f, results, 2)
end

println("\n" * "="^70)
println("SIMULATION PROFILING COMPLETE")
println("Results saved to scratch/profiling/simulation_baseline.json")
println("="^70)
