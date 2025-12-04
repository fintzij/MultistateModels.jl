"""
Debug script for MCEM Reversible Model with TVC

This script investigates the "sample path explosion" problem in reversible models
by stepping through the MCEM algorithm manually, replicating the exact setup
from test/longtest_mcem_tvc.jl "MCEM Reversible Model with TVC" test.

Key diagnostics:
- Path counts per subject per MCEM iteration
- ESS evolution across iterations
- Importance weight distributions
- Breaks out when paths exceed threshold
"""

using Pkg
Pkg.activate(".")

using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Printf
using ElasticArrays
using Optimization
using OptimizationOptimJL
import Optim

# Import internal functions for debugging
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_log_scale_params, SamplePath, draw_samplepath,
    build_tpm_mapping, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!,
    build_fbmats, ForwardFiltering!, loglik, nest_params, DrawSamplePaths!,
    mcem_mll, mcem_ase, SMPanelData, compute_markov_marginal_loglik, compute_loglik,
    viterbi_map_path, loglik!

using ExponentialUtilities
using ParetoSmooth

const RNG_SEED = 0xABCDEF01
const MAX_PATHS_DEBUG = 600  # Break out if paths exceed this
const N_SUBJECTS = 40  # Same as longtest

# ============================================================================
# Helper: Build TVC Panel Data with finer observation intervals
# ============================================================================

function build_tvc_panel_data_longtest(; n_subjects::Int, rng, n_intervals::Int=100, max_time::Float64=4.0)
    """
    Build panel data with many observation times and heterogeneous interval lengths.
    
    Args:
        n_subjects: Number of subjects
        rng: Random number generator
        n_intervals: Number of observation intervals per subject (default 100)
        max_time: Maximum observation time (default 4.0)
    """
    # Generate base observation times with HETEROGENEOUS intervals
    # Use random intervals drawn from exponential distribution, then normalize to max_time
    change_time = 1.5  # Treatment change point
    
    rows = []
    for subj in 1:n_subjects
        # Generate random interval lengths (exponential with mean 1)
        raw_intervals = randexp(rng, n_intervals)
        # Normalize so they sum to max_time
        raw_intervals = raw_intervals .* (max_time / sum(raw_intervals))
        # Convert to cumulative observation times
        subj_obs_times = cumsum(raw_intervals)
        # Ensure we end exactly at max_time (numerical precision)
        subj_obs_times[end] = max_time
        
        # Treatment assignment: half get treatment after change_time
        gets_treatment = rand(rng) < 0.5
        
        # Build time grid including change_time
        all_times = sort(unique([0.0; subj_obs_times; change_time]))
        
        for i in 1:(length(all_times)-1)
            t_start = all_times[i]
            t_stop = all_times[i+1]
            
            # Covariate: 0 before change_time, 1 after (if treated)
            x_val = (t_start >= change_time && gets_treatment) ? 1.0 : 0.0
            
            push!(rows, (id=subj, tstart=t_start, tstop=t_stop,
                         statefrom=1, stateto=1, obstype=2, x=x_val))
        end
    end
    
    return DataFrame(rows)
end

# ============================================================================
# Analyze sample path characteristics
# ============================================================================

function analyze_sample_paths(paths::Vector{SamplePath})
    n_paths = length(paths)
    
    if n_paths == 0
        return (n_paths = 0, mean_transitions = 0.0, max_transitions = 0, 
                n_unique_sequences = 0, top_sequences = [])
    end
    
    # Count transitions per path
    n_transitions = [length(p.times) - 1 for p in paths]
    
    # Count unique path structures (state sequences)
    state_seqs = [join(p.states, "-") for p in paths]
    unique_seqs = unique(state_seqs)
    seq_counts = [count(==(seq), state_seqs) for seq in unique_seqs]
    
    return (
        n_paths = n_paths,
        n_transitions = n_transitions,
        mean_transitions = mean(n_transitions),
        max_transitions = maximum(n_transitions),
        min_transitions = minimum(n_transitions),
        n_unique_sequences = length(unique_seqs),
        top_sequences = sort(collect(zip(unique_seqs, seq_counts)), by=x->-x[2])[1:min(10, length(unique_seqs))]
    )
end

# ============================================================================
# Manual MCEM iteration with full introspection
# ============================================================================

function mcem_iteration_step!(;
    model,
    surrogate,
    params_cur::Vector{Float64},
    samplepaths::Vector{Vector{SamplePath}},
    loglik_surrog::Vector{Vector{Float64}},
    loglik_target_cur::Vector{Vector{Float64}},
    loglik_target_prop::Vector{Vector{Float64}},
    _logImportanceWeights::Vector{Vector{Float64}},
    ImportanceWeights::Vector{Vector{Float64}},
    ess_cur::Vector{Float64},
    ess_target::Float64,
    max_sampling_effort::Int,
    npaths_additional::Int,
    tpm_book,
    hazmat_book,
    books,
    fbmats,
    absorbingstates,
    psis_pareto_k::Vector{Float64}
)
    """
    Execute a single MCEM iteration:
    1. Draw sample paths until ESS target met
    2. M-step optimization
    3. Update parameters
    
    Returns: params_new, mll_cur, mll_prop, ess_new, path_counts
    """
    nsubj = length(model.subjectindices)
    
    # Draw sample paths (this is the E-step)
    DrawSamplePaths!(model;
        ess_target = ess_target,
        ess_cur = ess_cur,
        max_sampling_effort = max_sampling_effort,
        samplepaths = samplepaths,
        loglik_surrog = loglik_surrog,
        loglik_target_prop = loglik_target_prop,
        loglik_target_cur = loglik_target_cur,
        _logImportanceWeights = _logImportanceWeights,
        ImportanceWeights = ImportanceWeights,
        tpm_book_surrogate = tpm_book,
        hazmat_book_surrogate = hazmat_book,
        books = books,
        npaths_additional = npaths_additional,
        params_cur = params_cur,
        surrogate = surrogate,
        psis_pareto_k = psis_pareto_k,
        fbmats = fbmats,
        absorbingstates = absorbingstates,
        # Phase-type infrastructure (nothing if not using)
        phasetype_surrogate = nothing,
        tpm_book_ph = nothing,
        hazmat_book_ph = nothing,
        fbmats_ph = nothing,
        emat_ph = nothing
    )
    
    # Current marginal log-likelihood
    mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
    
    # M-step: optimize
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights))
    sol = solve(prob, Optim.LBFGS())
    params_new = sol.u
    
    # Evaluate new log-likelihoods
    loglik!(params_new, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
    mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
    
    # Path counts
    path_counts = length.(samplepaths)
    
    return params_new, mll_cur, mll_prop, deepcopy(ess_cur), path_counts
end

# ============================================================================
# Full MCEM stepping with introspection
# ============================================================================

function debug_mcem_stepping(model; 
    maxiter::Int = 10,
    ess_target_initial::Float64 = 20.0,
    ess_increase::Float64 = 2.0,
    max_ess::Float64 = 300.0,
    max_sampling_effort::Int = 20,
    npaths_additional::Int = 10,
    max_paths_breakout::Int = MAX_PATHS_DEBUG,
    verbose::Bool = true)
    
    println("\n" * "=" ^ 70)
    println("DEBUG: Stepping Through MCEM Algorithm")
    println("=" ^ 70)
    
    nsubj = length(model.subjectindices)
    surrogate = model.markovsurrogate
    
    # Print parameter info
    params_cur = get_parameters_flat(model)
    surrogate_pars = get_log_scale_params(surrogate.parameters)
    
    println("\n--- Model Setup ---")
    println("Number of subjects: $nsubj")
    println("Target model hazards: $(typeof.(model.hazards))")
    println("Surrogate hazards: $(typeof.(surrogate.hazards))")
    println("\nInitial parameters (log-scale): $params_cur")
    println("Surrogate parameters (log-scale): $surrogate_pars")
    
    # Identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))
    println("Absorbing states: $absorbingstates")
    
    # Build infrastructure
    books = build_tpm_mapping(model.data)
    hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book = build_tpm_book(Float64, model.tmat, books[1])
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
    
    # Compute TPMs for surrogate
    for t in eachindex(books[1])
        compute_hazmat!(hazmat_book[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
        compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
    end
    
    # Build forward-backward matrices
    fbmats = any(model.data.obstype .> 2) ? build_fbmats(model) : nothing
    
    # Initialize containers
    ess_target = ess_target_initial
    ess_cur = zeros(nsubj)
    psis_pareto_k = zeros(nsubj)
    
    samplepaths = [sizehint!(Vector{SamplePath}(), Int(max_ess * max_sampling_effort)) for _ in 1:nsubj]
    loglik_surrog = [sizehint!(Vector{Float64}(), Int(max_ess * max_sampling_effort)) for _ in 1:nsubj]
    loglik_target_cur = [sizehint!(Vector{Float64}(), Int(max_ess * max_sampling_effort)) for _ in 1:nsubj]
    loglik_target_prop = [sizehint!(Vector{Float64}(), Int(max_ess * max_sampling_effort)) for _ in 1:nsubj]
    _logImportanceWeights = [sizehint!(Vector{Float64}(), Int(max_ess * max_sampling_effort)) for _ in 1:nsubj]
    ImportanceWeights = [sizehint!(Vector{Float64}(), Int(max_ess * max_sampling_effort)) for _ in 1:nsubj]
    
    # Traces
    mll_trace = Float64[]
    ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0)
    path_count_trace = ElasticArray{Int, 2}(undef, nsubj, 0)
    params_trace = ElasticArray{Float64, 2}(undef, length(params_cur), 0)
    
    # Compute normalizing constant
    NormConstantProposal = compute_markov_marginal_loglik(model, surrogate)
    println("Normalizing constant (Markov marginal LL): $(@sprintf("%.3f", NormConstantProposal))")
    
    # ========================================================================
    # Viterbi initialization (matching fit function)
    # ========================================================================
    println("\n--- Viterbi Initialization ---")
    for i in 1:nsubj
        subj_inds = model.subjectindices[i]
        if all(model.data.obstype[subj_inds] .== 1)
            continue
        end
        
        map_path = viterbi_map_path(i, model, tpm_book, hazmat_book, books[2], fbmats, absorbingstates)
        push!(samplepaths[i], map_path)
        
        ll_surrog = loglik(surrogate_pars, map_path, surrogate.hazards, model)
        target_pars = nest_params(params_cur, model.parameters)
        ll_target = loglik(target_pars, map_path, model.hazards, model)
        
        push!(loglik_surrog[i], ll_surrog)
        push!(loglik_target_cur[i], ll_target)
        push!(loglik_target_prop[i], 0.0)
        push!(_logImportanceWeights[i], ll_target - ll_surrog)
        push!(ImportanceWeights[i], 1.0)
        ess_cur[i] = 1.0
    end
    println("Viterbi initialization complete.")
    
    # ========================================================================
    # MCEM iterations
    # ========================================================================
    explosion_detected = false
    explosion_iter = -1
    explosion_subject = -1
    
    for iter in 1:maxiter
        println("\n" * "=" ^ 70)
        println("MCEM ITERATION $iter")
        println("=" ^ 70)
        
        println("\nE-step: Drawing sample paths...")
        println("  ESS target: $(@sprintf("%.1f", ess_target))")
        println("  Current path counts: $(length.(samplepaths))")
        println("  Current ESS: $([round(e, digits=1) for e in ess_cur])")
        
        # E-step: Draw sample paths
        DrawSamplePaths!(model;
            ess_target = ess_target,
            ess_cur = ess_cur,
            max_sampling_effort = max_sampling_effort,
            samplepaths = samplepaths,
            loglik_surrog = loglik_surrog,
            loglik_target_prop = loglik_target_prop,
            loglik_target_cur = loglik_target_cur,
            _logImportanceWeights = _logImportanceWeights,
            ImportanceWeights = ImportanceWeights,
            tpm_book_surrogate = tpm_book,
            hazmat_book_surrogate = hazmat_book,
            books = books,
            npaths_additional = npaths_additional,
            params_cur = params_cur,
            surrogate = surrogate,
            psis_pareto_k = psis_pareto_k,
            fbmats = fbmats,
            absorbingstates = absorbingstates,
            phasetype_surrogate = nothing,
            tpm_book_ph = nothing,
            hazmat_book_ph = nothing,
            fbmats_ph = nothing,
            emat_ph = nothing
        )
        
        # Check for path explosion
        path_counts = length.(samplepaths)
        println("\n  After E-step:")
        println("    Path counts: $path_counts")
        println("    Max path count: $(maximum(path_counts))")
        println("    ESS: $([round(e, digits=1) for e in ess_cur])")
        println("    Pareto-k: $([round(k, digits=2) for k in psis_pareto_k])")
        
        # Check for NaN ESS and analyze why
        nan_subjects = findall(isnan.(ess_cur))
        if !isempty(nan_subjects)
            println("\n  ⚠ Subjects with NaN ESS: $nan_subjects")
            for subj in nan_subjects[1:min(3, length(nan_subjects))]
                lw = _logImportanceWeights[subj]
                println("    Subject $subj log weights: min=$(@sprintf("%.4f", minimum(lw))), max=$(@sprintf("%.4f", maximum(lw))), std=$(@sprintf("%.6f", std(lw)))")
                # Check if all weights are nearly identical
                if std(lw) < 1e-10
                    println("      → All weights identical! Target = Surrogate")
                end
            end
        end
        
        max_paths = maximum(path_counts)
        if max_paths > max_paths_breakout
            explosion_subject = argmax(path_counts)
            explosion_iter = iter
            explosion_detected = true
            println("\n⚠⚠⚠ PATH EXPLOSION DETECTED ⚠⚠⚠")
            println("  Subject $explosion_subject has $max_paths paths (> $max_paths_breakout)")
            println("  Breaking out of MCEM loop!")
            break
        end
        
        # Detailed diagnostics for subjects with many paths
        for subj in 1:nsubj
            if path_counts[subj] > 100
                stats = analyze_sample_paths(samplepaths[subj])
                println("\n  Subject $subj detailed stats ($(path_counts[subj]) paths):")
                println("    Mean transitions: $(@sprintf("%.1f", stats.mean_transitions))")
                println("    Max transitions: $(stats.max_transitions)")
                println("    Unique sequences: $(stats.n_unique_sequences)")
                lw = _logImportanceWeights[subj]
                println("    Log weights: [$(@sprintf("%.2f", minimum(lw))), $(@sprintf("%.2f", maximum(lw)))]")
            end
        end
        
        # Current marginal log-likelihood
        mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
        println("\n  Current Q (weighted complete-data LL): $(@sprintf("%.3f", mll_cur))")
        
        # M-step
        println("\nM-step: Optimizing parameters...")
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights))
        sol = solve(prob, Optim.LBFGS())
        params_prop = sol.u
        
        println("  Old params: $([round(p, digits=3) for p in params_cur])")
        println("  New params: $([round(p, digits=3) for p in params_prop])")
        
        # Evaluate new log-likelihoods
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
        mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
        
        println("  Q change: $(@sprintf("%.3f", mll_cur)) -> $(@sprintf("%.3f", mll_prop)) (Δ=$(@sprintf("%.4f", mll_prop - mll_cur)))")
        
        # Update parameters
        params_cur = deepcopy(params_prop)
        for i in 1:nsubj
            loglik_target_cur[i] .= loglik_target_prop[i]
        end
        
        # Update ESS target (matching fit function logic)
        ess_target = min(ess_target * ess_increase, max_ess)
        
        # Record traces
        push!(mll_trace, mll_prop)
        append!(ess_trace, reshape(ess_cur, nsubj, 1))
        append!(path_count_trace, reshape(path_counts, nsubj, 1))
        append!(params_trace, reshape(params_cur, length(params_cur), 1))
        
        println("\nNext ESS target: $(@sprintf("%.1f", ess_target))")
    end
    
    # ========================================================================
    # Summary
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("MCEM DEBUG SUMMARY")
    println("=" ^ 70)
    
    println("\nExplosion detected: $explosion_detected")
    if explosion_detected
        println("  Iteration: $explosion_iter")
        println("  Subject: $explosion_subject")
        println("  Path count: $(length(samplepaths[explosion_subject]))")
        
        # Detailed analysis of explosion subject
        stats = analyze_sample_paths(samplepaths[explosion_subject])
        println("\nExplosion subject analysis:")
        println("  Total paths: $(stats.n_paths)")
        println("  Mean transitions: $(@sprintf("%.1f", stats.mean_transitions))")
        println("  Max transitions: $(stats.max_transitions)")
        println("  Unique sequences: $(stats.n_unique_sequences)")
        println("  Top sequences:")
        for (i, (seq, cnt)) in enumerate(stats.top_sequences[1:min(5, length(stats.top_sequences))])
            pct = 100 * cnt / stats.n_paths
            println("    $i. '$seq': $cnt ($(@sprintf("%.1f", pct))%)")
        end
        
        # Subject data
        subj_inds = model.subjectindices[explosion_subject]
        subj_dat = model.data[subj_inds, :]
        println("\n  Subject data:")
        println("    Time span: $(first(subj_dat.tstart)) to $(last(subj_dat.tstop))")
        println("    Observed transitions: $(sum(subj_dat.statefrom .!= subj_dat.stateto))")
        println("    Start state: $(first(subj_dat.statefrom)), End state: $(last(subj_dat.stateto))")
    end
    
    println("\nFinal statistics:")
    println("  Iterations completed: $(length(mll_trace))")
    println("  Final path counts: $(length.(samplepaths))")
    println("  Total paths: $(sum(length.(samplepaths)))")
    println("  Final ESS: $([round(e, digits=1) for e in ess_cur])")
    println("  Final params: $([round(p, digits=3) for p in params_cur])")
    
    return (
        explosion = explosion_detected,
        explosion_iter = explosion_iter,
        explosion_subject = explosion_subject,
        samplepaths = samplepaths,
        mll_trace = mll_trace,
        ess_trace = ess_trace,
        path_count_trace = path_count_trace,
        params_trace = params_trace,
        final_params = params_cur
    )
end

# ============================================================================
# Main debug analysis - with finer observation intervals
# ============================================================================

function run_debug_analysis()
    rng = Random.MersenneTwister(RNG_SEED + 7)  # Same seed as longtest
    
    println("=" ^ 80)
    println("DEBUG: MCEM Reversible Model with TVC - Path Explosion Investigation")
    println("=" ^ 80)
    println("\nUsing VERY FINE heterogeneous observation intervals")
    
    n_subj = N_SUBJECTS  # 40, same as longtest
    n_intervals = 100    # 100 intervals with heterogeneous lengths
    max_time = 4.0
    mean_interval = max_time / n_intervals
    
    println("\nTest setup (VERY FINE HETEROGENEOUS INTERVALS):")
    println("  Subjects: $n_subj")
    println("  Observation intervals: $n_intervals (heterogeneous lengths)")
    println("  Mean interval length: $(@sprintf("%.3f", mean_interval)) time units")
    println("  Interval distribution: Exponential (varying per subject)")
    println("  Max time: $max_time")
    println("  Treatment change time: 1.5")
    
    # ========================================================================
    # Build panel data with very fine heterogeneous intervals
    # ========================================================================
    panel_data = build_tvc_panel_data_longtest(n_subjects=n_subj, rng=rng, 
                                                n_intervals=n_intervals, max_time=max_time)
    
    println("\nPanel data created:")
    println("  Total rows: $(nrow(panel_data))")
    println("  Rows per subject: ~$(nrow(panel_data) ÷ n_subj)")
    
    # Show sample of observation times for first subject
    subj1_data = filter(row -> row.id == 1, panel_data)
    println("\n  Subject 1 interval lengths:")
    interval_lens = subj1_data.tstop .- subj1_data.tstart
    println("    Min: $(@sprintf("%.3f", minimum(interval_lens)))")
    println("    Max: $(@sprintf("%.3f", maximum(interval_lens)))")
    println("    Mean: $(@sprintf("%.3f", mean(interval_lens)))")
    
    # ========================================================================
    # Create and simulate with exact longtest parameters
    # ========================================================================
    println("\n" * "=" ^ 80)
    println("SIMULATION: Exact longtest parameters")
    println("=" ^ 80)
    
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h21_sim = Hazard(@formula(0 ~ x), "wei", 2, 1)
    model_sim = multistatemodel(h12_sim, h21_sim; data=panel_data, surrogate=:markov)
    
    # Exact longtest parameters
    set_parameters!(model_sim, (
        h12 = [log(1.0), log(0.1), 0.3],   # shape=1.0, scale=0.1, beta=0.3
        h21 = [log(1.0), log(0.15), 0.2]   # shape=1.0, scale=0.15, beta=0.2
    ))
    
    println("\nSimulation parameters (EXACT LONGTEST):")
    println("  h12: shape=1.0, scale=0.1 (mean sojourn ~10), beta=0.3")
    println("  h21: shape=1.0, scale=0.15 (mean sojourn ~6.7), beta=0.2")
    
    # Simulate
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    simulated_data = sim_result[1, 1]
    
    # Count transitions
    trans = combine(groupby(simulated_data, :id)) do gdf
        (n_trans = sum(gdf.statefrom .!= gdf.stateto),)
    end
    println("\nSimulated data transitions:")
    println("  Total: $(sum(trans.n_trans))")
    println("  Max per subject: $(maximum(trans.n_trans))")
    println("  Subjects with 0 transitions: $(sum(trans.n_trans .== 0))")
    println("  Subjects with 1+ transitions: $(sum(trans.n_trans .>= 1))")
    println("  Subjects with 2+ transitions: $(sum(trans.n_trans .>= 2))")
    
    # ========================================================================
    # Create fitting model (parameters start at defaults)
    # ========================================================================
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h21_fit = Hazard(@formula(0 ~ x), "wei", 2, 1)
    model_fit = multistatemodel(h12_fit, h21_fit; data=simulated_data, surrogate=:markov)
    
    println("\nFitting model initial parameters (defaults):")
    println("  $(get_parameters_flat(model_fit))")
    
    # ========================================================================
    # Run MCEM stepping with settings
    # ========================================================================
    println("\n" * "=" ^ 80)
    println("MCEM STEPPING WITH 100 HETEROGENEOUS INTERVALS")
    println("=" ^ 80)
    
    results = debug_mcem_stepping(model_fit;
        maxiter = 10,              # Enough to see if explosion occurs
        ess_target_initial = 20.0, # Same as longtest
        ess_increase = 2.0,
        max_ess = 300.0,           # Same as longtest
        max_sampling_effort = 20,
        npaths_additional = 10,
        max_paths_breakout = MAX_PATHS_DEBUG,
        verbose = true
    )
    
    println("\n" * "=" ^ 80)
    println("DEBUG COMPLETE")
    println("=" ^ 80)
    
    return results
end

# Run
results = run_debug_analysis()
