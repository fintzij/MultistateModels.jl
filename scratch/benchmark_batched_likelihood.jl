# =============================================================================
# Benchmark: Batched vs Sequential Likelihood Computation
# =============================================================================
#
# This script compares the performance of:
# 1. loglik_exact (sequential, path-by-path)
# 2. loglik_exact_batched (hazard-centric, batched)
#
# The batched version should be faster when:
# - Many paths (MCEM scenarios with hundreds/thousands of imputed paths)
# - Overhead of path iteration dominates
#
# Usage:
#   julia --project=. scratch/benchmark_batched_likelihood.jl

using MultistateModels
using DataFrames
using Random
using BenchmarkTools
using ArraysOfArrays: flatview
using MultistateModels: ExactData, loglik_exact, loglik_exact_batched, cache_path_data,
                         SMPanelData, loglik_semi_markov!, loglik_semi_markov_batched!

# =============================================================================
# Helper: Generate test data
# =============================================================================

"""
Generate a multi-subject illness-death dataset with exact observations.
"""
function generate_illness_death_data(; n_subjects=100, max_transitions=5, seed=12345)
    Random.seed!(seed)
    
    rows = NamedTuple[]
    id = 0
    
    for _ in 1:n_subjects
        id += 1
        state = 1  # Start in state 1 (healthy)
        t = 0.0
        age = rand(30:70)
        trt = rand([0, 1])
        
        for _ in 1:max_transitions
            if state == 3  # Absorbing state (death)
                break
            end
            
            # Sojourn time (exponential with rate depending on state)
            rate = state == 1 ? 0.3 : 0.5
            sojourn = rand() * 5 / rate
            
            # Next state (probabilistic)
            if state == 1
                next_state = rand() < 0.6 ? 2 : 3  # illness or death
            else  # state == 2
                next_state = 3  # death
            end
            
            push!(rows, (
                id = id,
                tstart = t,
                tstop = t + sojourn,
                statefrom = state,
                stateto = next_state,
                obstype = 1,  # exact
                age = age,
                trt = trt
            ))
            
            t += sojourn
            state = next_state
        end
    end
    
    return DataFrame(rows)
end

# =============================================================================
# Benchmark Suite
# =============================================================================

function run_benchmarks()
    println("="^70)
    println("Batched vs Sequential Likelihood Benchmark")
    println("="^70)
    println()
    
    # Test configurations
    configs = [
        (n_subjects=10, name="Small (10 subjects)"),
        (n_subjects=50, name="Medium (50 subjects)"),
        (n_subjects=100, name="Large (100 subjects)"),
        (n_subjects=500, name="Very Large (500 subjects)"),
    ]
    
    results = DataFrame(
        config = String[],
        n_subjects = Int[],
        n_paths = Int[],
        n_intervals = Int[],
        seq_time_ms = Float64[],
        bat_time_ms = Float64[],
        speedup = Float64[],
        values_match = Bool[]
    )
    
    for (n_subjects, name) in configs
        println("\n--- $name ---")
        
        # Generate data
        dat = generate_illness_death_data(n_subjects=n_subjects)
        
        # Define hazards (illness-death model)
        h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)  # healthy -> ill
        h13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)         # healthy -> death
        h23 = Hazard(@formula(0 ~ 1 + age + trt), "gom", 2, 3)  # ill -> death
        
        # Build model
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (
            h12 = [log(0.1), log(1.0), 0.01, 0.3],
            h13 = [log(0.05), 0.02],
            h23 = [log(0.2), 0.03, 0.02, 0.2]
        ))
        
        # Extract paths and create ExactData
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        # Count intervals
        cached = cache_path_data(paths, model)
        n_intervals = sum(nrow(cpd.df) for cpd in cached)
        
        println("  Paths: $(length(paths)), Total intervals: $n_intervals")
        
        # Verify correctness
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact_batched(pars, exact_data; neg=false)
        values_match = isapprox(ll_seq, ll_bat, rtol=1e-12)
        
        if !values_match
            @warn "Values don't match! seq=$ll_seq, bat=$ll_bat"
        end
        
        # Benchmark
        println("  Benchmarking sequential...")
        b_seq = @benchmark loglik_exact($pars, $exact_data; neg=false) samples=50 evals=3
        
        println("  Benchmarking batched...")
        b_bat = @benchmark loglik_exact_batched($pars, $exact_data; neg=false) samples=50 evals=3
        
        # Extract times in milliseconds
        seq_time = median(b_seq).time / 1e6
        bat_time = median(b_bat).time / 1e6
        speedup = seq_time / bat_time
        
        println("  Sequential: $(round(seq_time, digits=3)) ms")
        println("  Batched:    $(round(bat_time, digits=3)) ms")
        println("  Speedup:    $(round(speedup, digits=2))x")
        
        push!(results, (
            config = name,
            n_subjects = n_subjects,
            n_paths = length(paths),
            n_intervals = n_intervals,
            seq_time_ms = seq_time,
            bat_time_ms = bat_time,
            speedup = speedup,
            values_match = values_match
        ))
    end
    
    println("\n")
    println("="^70)
    println("Summary")
    println("="^70)
    println(results)
    
    return results
end

# =============================================================================
# MCEM-style Benchmark: Multiple paths per subject
# =============================================================================

"""
Test scaling with larger number of subjects (not MCEM-style).
This is the scenario where batched likelihood provides actual benefit.
"""
function run_scaling_benchmark()
    println("\n")
    println("="^70)
    println("Scaling Benchmark: Increasing Number of Subjects")
    println("="^70)
    println()
    
    results = DataFrame(
        n_subjects = Int[],
        n_paths = Int[],
        n_intervals = Int[],
        seq_time_ms = Float64[],
        bat_time_ms = Float64[],
        speedup = Float64[],
        values_match = Bool[]
    )
    
    for n_subj in [10, 25, 50, 100, 200, 500]
        println("\n--- $n_subj subjects ---")
        
        dat = generate_illness_death_data(n_subjects=n_subj)
        
        h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)
        h13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1 + age + trt), "gom", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (
            h12 = [log(0.1), log(1.0), 0.01, 0.3],
            h13 = [log(0.05), 0.02],
            h23 = [log(0.2), 0.03, 0.02, 0.2]
        ))
        
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = flatview(model.parameters)
        
        # Count intervals
        cached = cache_path_data(paths, model)
        n_intervals = sum(nrow(cpd.df) for cpd in cached)
        
        println("  Paths: $(length(paths)), Intervals: $n_intervals")
        
        # Verify correctness
        ll_seq = loglik_exact(pars, exact_data; neg=false)
        ll_bat = loglik_exact_batched(pars, exact_data; neg=false)
        values_match = isapprox(ll_seq, ll_bat, rtol=1e-12)
        
        if !values_match
            @warn "Values don't match! seq=$ll_seq, bat=$ll_bat"
        end
        
        # Benchmark
        println("  Benchmarking...")
        b_seq = @benchmark loglik_exact($pars, $exact_data; neg=false) samples=30 evals=2
        b_bat = @benchmark loglik_exact_batched($pars, $exact_data; neg=false) samples=30 evals=2
        
        seq_time = median(b_seq).time / 1e6
        bat_time = median(b_bat).time / 1e6
        speedup = seq_time / bat_time
        
        println("  Sequential: $(round(seq_time, digits=3)) ms")
        println("  Batched:    $(round(bat_time, digits=3)) ms")
        println("  Speedup:    $(round(speedup, digits=2))x")
        println("  Values match: $values_match")
        
        push!(results, (
            n_subjects = n_subj,
            n_paths = length(paths),
            n_intervals = n_intervals,
            seq_time_ms = seq_time,
            bat_time_ms = bat_time,
            speedup = speedup,
            values_match = values_match
        ))
    end
    
    println("\n")
    println("="^70)
    println("Scaling Summary")
    println("="^70)
    println(results)
    
    return results
end

# =============================================================================
# Memory Allocation Comparison
# =============================================================================

function compare_allocations()
    println("\n")
    println("="^70)
    println("Memory Allocation Comparison")
    println("="^70)
    
    # Medium-sized test case
    dat = generate_illness_death_data(n_subjects=50)
    h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)
    h13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
    h23 = Hazard(@formula(0 ~ 1 + age + trt), "gom", 2, 3)
    
    model = multistatemodel(h12, h13, h23; data = dat)
    set_parameters!(model, (
        h12 = [log(0.1), log(1.0), 0.01, 0.3],
        h13 = [log(0.05), 0.02],
        h23 = [log(0.2), 0.03, 0.02, 0.2]
    ))
    
    paths = MultistateModels.extract_paths(model)
    exact_data = ExactData(model, paths)
    pars = flatview(model.parameters)
    
    println("\nSequential (loglik_exact):")
    @time loglik_exact(pars, exact_data; neg=false)
    
    println("\nBatched (loglik_exact_batched):")
    @time loglik_exact_batched(pars, exact_data; neg=false)
end

"""
Benchmark MCEM-style computation with loglik_semi_markov vs loglik_semi_markov_batched.
"""
function run_mcem_benchmark()
    
    println("\n")
    println("="^70)
    println("MCEM Benchmark: loglik_semi_markov! vs loglik_semi_markov_batched!")
    println("="^70)
    println()
    
    results = DataFrame(
        n_subjects = Int[],
        n_paths_per_subj = Int[],
        total_paths = Int[],
        seq_time_ms = Float64[],
        bat_time_ms = Float64[],
        speedup = Float64[],
        values_match = Bool[]
    )
    
    # Test different configurations
    configs = [
        (n_subj=10, n_paths=5),
        (n_subj=10, n_paths=25),
        (n_subj=25, n_paths=10),
        (n_subj=25, n_paths=50),
        (n_subj=50, n_paths=20),
        (n_subj=100, n_paths=10),
    ]
    
    for (n_subj, n_paths_per) in configs
        println("\n--- $n_subj subjects, $n_paths_per paths each ---")
        
        dat = generate_illness_death_data(n_subjects=n_subj)
        
        h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)
        h13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1 + age + trt), "gom", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data = dat)
        set_parameters!(model, (
            h12 = [log(0.1), log(1.0), 0.01, 0.3],
            h13 = [log(0.05), 0.02],
            h23 = [log(0.2), 0.03, 0.02, 0.2]
        ))
        
        base_paths = MultistateModels.extract_paths(model)
        
        # Create nested paths structure
        nested_paths = [
            [deepcopy(base_paths[i]) for _ in 1:n_paths_per]
            for i in 1:n_subj
        ]
        importance_weights = [ones(n_paths_per) for _ in 1:n_subj]
        
        smpanel = SMPanelData(model, nested_paths, importance_weights)
        pars = flatview(model.parameters)
        
        total_paths = n_subj * n_paths_per
        println("  Total paths: $total_paths")
        
        # Initialize log-likelihood structures
        logliks_seq = [zeros(n_paths_per) for _ in 1:n_subj]
        logliks_bat = [zeros(n_paths_per) for _ in 1:n_subj]
        
        # Verify correctness
        loglik_semi_markov!(pars, logliks_seq, smpanel)
        loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
        
        values_match = all(
            isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
            for i in 1:n_subj for j in 1:n_paths_per
        )
        
        if !values_match
            @warn "Values don't match!"
        end
        
        # Benchmark - need to re-zero the logliks each time
        println("  Benchmarking...")
        
        b_seq = @benchmark begin
            for i in eachindex($logliks_seq)
                fill!($logliks_seq[i], 0.0)
            end
            loglik_semi_markov!($pars, $logliks_seq, $smpanel)
        end samples=30 evals=2
        
        b_bat = @benchmark begin
            for i in eachindex($logliks_bat)
                fill!($logliks_bat[i], 0.0)
            end
            loglik_semi_markov_batched!($pars, $logliks_bat, $smpanel)
        end samples=30 evals=2
        
        seq_time = median(b_seq).time / 1e6
        bat_time = median(b_bat).time / 1e6
        speedup = seq_time / bat_time
        
        println("  Sequential: $(round(seq_time, digits=3)) ms")
        println("  Batched:    $(round(bat_time, digits=3)) ms")
        println("  Speedup:    $(round(speedup, digits=2))x")
        println("  Values match: $values_match")
        
        push!(results, (
            n_subjects = n_subj,
            n_paths_per_subj = n_paths_per,
            total_paths = total_paths,
            seq_time_ms = seq_time,
            bat_time_ms = bat_time,
            speedup = speedup,
            values_match = values_match
        ))
    end
    
    println("\n")
    println("="^70)
    println("MCEM Summary")
    println("="^70)
    println(results)
    
    return results
end

# =============================================================================
# Run if executed directly
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_benchmarks()
    scaling_results = run_scaling_benchmark()
    mcem_results = run_mcem_benchmark()
    compare_allocations()
end
