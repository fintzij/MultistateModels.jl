# =============================================================================
# Timing Test for Alpha Learning Optimization
# =============================================================================
#
# This script benchmarks the performance of PIJCV with different learn_alpha settings.
# Run with: julia --project=.. timing_test_alpha_learning.jl
#
# Target: learn_alpha=true should be within 2x of standard PIJCV runtime
# =============================================================================

using MultistateModels
using DataFrames
using Random
using Statistics

# Reproducibility
Random.seed!(20260126)

println("="^70)
println("ALPHA LEARNING PERFORMANCE BENCHMARK")
println("="^70)

# =============================================================================
# Generate Test Data (illness-death, 500 subjects)
# =============================================================================

# True hazards for simulation
true_h12(t) = t > 0 ? 0.3 * sqrt(t) : 0.3 * sqrt(0.01)
true_h13(t) = 0.1 + 0.02 * t
true_h23(t) = 0.4 * exp(-0.1 * t)

# Cumulative hazards
true_H12(t) = t > 0 ? 0.3 * (2/3) * t^1.5 : 0.0
true_H13(t) = 0.1 * t + 0.01 * t^2
true_H23(t) = t > 0 ? 4.0 * (1 - exp(-0.1 * t)) : 0.0

function find_event_time(t_start, t_max, u, H_func; tol=1e-6, maxiter=100)
    target = -log(u)
    
    if H_func(t_max - t_start) < target
        return t_max + 1.0
    end
    
    lo, hi = t_start, t_max
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        if H_func(mid - t_start) < target
            lo = mid
        else
            hi = mid
        end
        if hi - lo < tol
            break
        end
    end
    return (lo + hi) / 2
end

function simulate_subject(max_time::Float64; rng=Random.GLOBAL_RNG)
    records = NamedTuple[]
    current_state = 1
    current_time = 0.0
    
    while current_state != 3 && current_time < max_time
        if current_state == 1
            u = rand(rng)
            t_event = find_event_time(current_time, max_time, u, 
                                       t -> true_H12(t - current_time) + true_H13(t - current_time))
            
            if t_event >= max_time
                push!(records, (tstart=current_time, tstop=max_time, statefrom=1, stateto=1, obstype=2))
                break
            end
            
            h12_t = true_h12(t_event - current_time)
            h13_t = true_h13(t_event - current_time)
            prob_12 = h12_t / (h12_t + h13_t)
            
            if rand(rng) < prob_12
                push!(records, (tstart=current_time, tstop=t_event, statefrom=1, stateto=2, obstype=1))
                current_state = 2
            else
                push!(records, (tstart=current_time, tstop=t_event, statefrom=1, stateto=3, obstype=1))
                current_state = 3
            end
            current_time = t_event
            
        elseif current_state == 2
            u = rand(rng)
            t_sojourn_start = current_time
            t_event = find_event_time(current_time, max_time, u, 
                                       t -> true_H23(t - t_sojourn_start))
            
            if t_event >= max_time
                push!(records, (tstart=current_time, tstop=max_time, statefrom=2, stateto=2, obstype=2))
                break
            end
            
            push!(records, (tstart=current_time, tstop=t_event, statefrom=2, stateto=3, obstype=1))
            current_state = 3
            current_time = t_event
        end
    end
    
    return records
end

function simulate_cohort(n_subjects, max_time; seed=20260126)
    rng = Random.MersenneTwister(seed)
    
    rows = NamedTuple{(:id, :tstart, :tstop, :statefrom, :stateto, :obstype), 
                       Tuple{Int, Float64, Float64, Int, Int, Int}}[]
    
    for i in 1:n_subjects
        records = simulate_subject(max_time; rng=rng)
        for r in records
            push!(rows, (id=i, tstart=r.tstart, tstop=r.tstop, 
                        statefrom=r.statefrom, stateto=r.stateto, obstype=r.obstype))
        end
    end
    
    return DataFrame(rows)
end

println("\n1. Generating test data...")
n_subjects = 500
max_time = 10.0
data = simulate_cohort(n_subjects, max_time)

println("   Generated $n_subjects subjects")
println("   Unique transitions: $(nrow(data))")

# =============================================================================
# Build Models
# =============================================================================

println("\n2. Building models...")

hazards = [
    Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=10.0),
    Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=10.0),
    Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=10.0)
]

model = multistatemodel(hazards...; data=data)

println("   Model built successfully")

# =============================================================================
# Timing Benchmarks
# =============================================================================

println("\n3. Running timing benchmarks...")
println("-"^70)

# Warm-up compilation run
println("\n   Warm-up run (compilation)...")
_ = fit(model; 
        penalty=SplinePenalty(),
        select_lambda=:pijcv,
        verbose=false)

# Benchmark: Standard PIJCV (uniform weighting, no alpha)
println("\n   [A] Standard PIJCV (uniform weighting)...")
t_standard = @elapsed fitted_standard = fit(model; 
    penalty=SplinePenalty(),  # Uniform weighting (default)
    select_lambda=:pijcv,
    verbose=false)
println("       Time: $(round(t_standard, digits=2))s")

# Benchmark: At-risk weighting with fixed alpha=1
println("\n   [B] PIJCV with at-risk weighting (α=1 fixed)...")
t_atrisk_fixed = @elapsed fitted_atrisk_fixed = fit(model;
    penalty=SplinePenalty(adaptive_weight=:atrisk, alpha=1.0, learn_alpha=false),
    select_lambda=:pijcv,
    verbose=false)
println("       Time: $(round(t_atrisk_fixed, digits=2))s")

# Benchmark: At-risk weighting with learned alpha
println("\n   [C] PIJCV with learned α (current implementation)...")
t_learn_alpha = @elapsed fitted_learn_alpha = fit(model;
    penalty=SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true),
    select_lambda=:pijcv,
    verbose=true)  # Verbose to see alpha iterations
println("       Time: $(round(t_learn_alpha, digits=2))s")

# =============================================================================
# Results Summary
# =============================================================================

println("\n" * "="^70)
println("TIMING RESULTS SUMMARY")
println("="^70)

println("\n   Method                        Time (s)    Relative to Standard")
println("   " * "-"^62)
println("   [A] Standard PIJCV            $(lpad(round(t_standard, digits=2), 8))    1.00x")
println("   [B] At-risk (α=1 fixed)       $(lpad(round(t_atrisk_fixed, digits=2), 8))    $(round(t_atrisk_fixed/t_standard, digits=2))x")
println("   [C] At-risk (learn α)         $(lpad(round(t_learn_alpha, digits=2), 8))    $(round(t_learn_alpha/t_standard, digits=2))x")

println("\n   TARGET: [C] should be ≤ 2.0x of [A]")
target_met = t_learn_alpha <= 2.0 * t_standard
println("   STATUS: $(target_met ? "✓ TARGET MET" : "✗ TARGET NOT MET")")

# Compare fitted values
println("\n" * "="^70)
println("FITTED VALUES COMPARISON")
println("="^70)

# Get smoothing parameters
λ_standard = fitted_standard.smoothing_parameters
λ_atrisk = fitted_atrisk_fixed.smoothing_parameters
λ_learn = fitted_learn_alpha.smoothing_parameters

println("\n   Smoothing parameters (λ):")
println("   [A] Standard:     $(round.(values(λ_standard), sigdigits=3))")
println("   [B] α=1 fixed:    $(round.(values(λ_atrisk), sigdigits=3))")
println("   [C] Learned α:    $(round.(values(λ_learn), sigdigits=3))")

# Get EDF
edf_standard = fitted_standard.edf
edf_atrisk = fitted_atrisk_fixed.edf  
edf_learn = fitted_learn_alpha.edf

println("\n   Effective degrees of freedom (EDF):")
println("   [A] Standard:     $(round(edf_standard.total, digits=2))")
println("   [B] α=1 fixed:    $(round(edf_atrisk.total, digits=2))")
println("   [C] Learned α:    $(round(edf_learn.total, digits=2))")

println("\n" * "="^70)
