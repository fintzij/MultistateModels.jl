# Profile MCEM Inference Performance
# Run with: julia --project=. scratch/profiling/profile_mcem.jl

using MultistateModels
using MultistateModels: loglik, SMPanelData, MPanelData,
                        build_tpm_mapping, safe_unflatten,
                        DrawSamplePaths!, ComputeImportanceWeightsESS!
using Profile
using BenchmarkTools
using Printf
using JSON
using Dates

# Include fixtures
include("setup_profiling_fixtures.jl")
using .ProfilingFixtures

println("="^70)
println("MCEM PROFILING - Phase 0 Baseline")
println("="^70)
println("Date: ", Dates.now())
println()

# Store results
results = Dict{String, Any}()

# -----------------------------------------------------------------------------
# Full MCEM fit (limited iterations)
# -----------------------------------------------------------------------------
println("\n--- Full MCEM Fit (3 iterations) ---\n")

mcem_fit_results = Dict{String, Any}()

for (name, create_fn) in [
    ("2-state Semi-Markov (n=30)", () -> create_semimarkov_2state(nsubj=30)),
]
    model = create_fn()
    set_surrogate!(model)  # Required for MCEM
    
    println("Testing: $name")
    
    # Time full fit with limited iterations
    t_start = time()
    fitted = fit(model; maxiter=3, verbose=true, tolerance=1e-8)
    t_elapsed = time() - t_start
    
    @printf("  Total time: %.2f s\n", t_elapsed)
    @printf("  Time per iteration: %.2f s\n", t_elapsed/3)
    
    mcem_fit_results[name] = Dict(
        "total_time_s" => t_elapsed,
        "time_per_iter_s" => t_elapsed/3,
        "iterations" => 3
    )
end

results["mcem_fit"] = mcem_fit_results

# -----------------------------------------------------------------------------
# Profile: MCEM iteration breakdown
# -----------------------------------------------------------------------------
println("\n--- Profiling MCEM Hot Path ---\n")

model = create_semimarkov_2state(nsubj=30)
set_surrogate!(model)  # Required for MCEM

println("Collecting profile data (3 iterations)...")

Profile.clear()
@profile fit(model; maxiter=3, verbose=false, tolerance=1e-8)

println("\nTop 30 functions by time:")
Profile.print(maxdepth=30, mincount=30, noisefloor=2.0)

# -----------------------------------------------------------------------------
# Scaling with nsubj
# -----------------------------------------------------------------------------
println("\n--- Scaling with nsubj (single MCEM iteration) ---\n")

scaling_results = Dict{String, Any}()

for nsubj in [20, 50, 100]
    model = create_semimarkov_2state(nsubj=nsubj)
    set_surrogate!(model)  # Required for MCEM
    
    t_start = time()
    fitted = fit(model; maxiter=1, verbose=false, tolerance=1e-8)
    t_elapsed = time() - t_start
    
    @printf("nsubj=%3d: %.2f s\n", nsubj, t_elapsed)
    
    scaling_results["nsubj_$nsubj"] = Dict(
        "time_s" => t_elapsed
    )
end

results["scaling_nsubj"] = scaling_results

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
results["metadata"] = Dict(
    "date" => string(Dates.now()),
    "julia_version" => string(VERSION),
    "phase" => "0_baseline"
)

open("scratch/profiling/mcem_baseline.json", "w") do f
    JSON.print(f, results, 2)
end

println("\n" * "="^70)
println("MCEM PROFILING COMPLETE")
println("Results saved to scratch/profiling/mcem_baseline.json")
println("="^70)
