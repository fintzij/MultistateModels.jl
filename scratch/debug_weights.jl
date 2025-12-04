"""
Diagnostic script to understand importance weight behavior in MCEM.
Tests whether the Markov surrogate is a good proposal for the semi-Markov target.
"""

using Pkg
Pkg.activate(".")

using Random
Random.seed!(0xABCDEF01)

using MultistateModels
using DataFrames
using Statistics
using LinearAlgebra
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_log_scale_params, SamplePath, draw_samplepath,
    build_tpm_mapping, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!,
    build_fbmats, ForwardFiltering!, loglik, nest_params, DrawSamplePaths!,
    mcem_mll, mcem_ase, SMPanelData, compute_markov_marginal_loglik, compute_loglik,
    viterbi_map_path, loglik!, ComputeImportanceWeightsESS!

using ExponentialUtilities
using ParetoSmooth

println("=" ^ 70)
println("MCEM Weight Diagnostics")
println("=" ^ 70)

# ============================================================================
# Test 1: Simple model with exponential (shape=1) - should be perfect proposal
# ============================================================================
println("\n--- Test 1: Exponential target (shape=1) ---")
println("Expected: Markov surrogate should be nearly perfect proposal")

n_subj = 10
max_time = 4.0

# Build panel data - uniform intervals, no TVC
rows = []
for subj in 1:n_subj
    for t in 0.0:0.2:max_time-0.2
        push!(rows, (id=subj, tstart=t, tstop=t+0.2, statefrom=1, stateto=1, obstype=2))
    end
end
panel_data = DataFrame(rows)

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
model_sim = multistatemodel(h12, h21; data=panel_data, surrogate=:markov)

# True parameters: exponential (shape=1)
set_parameters!(model_sim, (
    h12 = [log(1.0), log(0.5)],  # shape=1, scale=0.5
    h21 = [log(1.0), log(0.3)]   # shape=1, scale=0.3
))

println("  Simulating data...")
sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
simulated_data = sim_result[1, 1]

trans = combine(groupby(simulated_data, :id)) do gdf
    (n_trans = sum(gdf.statefrom .!= gdf.stateto),)
end
println("  Transitions per subject: ", trans.n_trans)

# Create fitting model
model_fit = multistatemodel(h12, h21; data=simulated_data, surrogate=:markov)
println("  Initial params (log-scale): ", get_parameters_flat(model_fit))

surrogate = model_fit.markovsurrogate
surrogate_pars = get_log_scale_params(surrogate.parameters)
println("  Surrogate params: ", surrogate_pars)

# Build sampling infrastructure
absorbingstates = findall(map(x -> all(x .== 0), eachrow(model_fit.tmat)))
books = build_tpm_mapping(model_fit.data)
hazmat_book = build_hazmat_book(Float64, model_fit.tmat, books[1])
tpm_book = build_tpm_book(Float64, model_fit.tmat, books[1])
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

for t in eachindex(books[1])
    compute_hazmat!(hazmat_book[t], surrogate_pars, surrogate.hazards, books[1][t], model_fit.data)
    compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
end

fbmats = any(model_fit.data.obstype .> 2) ? build_fbmats(model_fit) : nothing

# Sample paths and compute weights
params_cur = get_parameters_flat(model_fit)
nsubj = length(model_fit.subjectindices)

function sample_and_analyze(model_fit, surrogate, params_cur, tpm_book, hazmat_book, books, fbmats, absorbingstates, n_paths=100)
    nsubj = length(model_fit.subjectindices)
    surrogate_pars = get_log_scale_params(surrogate.parameters)
    
    results = []
    
    for subj in 1:min(3, nsubj)  # Analyze first 3 subjects
        subj_inds = model_fit.subjectindices[subj]
        subj_dat = view(model_fit.data, subj_inds, :)
        
        if any(subj_dat.obstype .∉ Ref([1,2]))
            subj_tpm_map = view(books[2], subj_inds, :)
            subj_emat = view(model_fit.emat, subj_inds, :)
            ForwardFiltering!(fbmats[subj], subj_dat, tpm_book, subj_tpm_map, subj_emat)
        end
        
        loglik_surrog = Float64[]
        loglik_target = Float64[]
        log_weights = Float64[]
        
        for j in 1:n_paths
            path = draw_samplepath(subj, model_fit, tpm_book, hazmat_book, books[2], fbmats, absorbingstates)
            
            ll_surrog = loglik(surrogate_pars, path, surrogate.hazards, model_fit)
            push!(loglik_surrog, ll_surrog)
            
            target_pars = nest_params(params_cur, model_fit.parameters)
            ll_target = loglik(target_pars, path, model_fit.hazards, model_fit)
            push!(loglik_target, ll_target)
            
            push!(log_weights, ll_target - ll_surrog)
        end
        
        # Compute ESS
        w = exp.(log_weights .- maximum(log_weights))
        w_norm = w ./ sum(w)
        simple_ess = 1.0 / sum(w_norm.^2)
        
        # PSIS
        psis_ess = NaN
        pareto_k = NaN
        try
            logw = reshape(log_weights, 1, length(log_weights), 1)
            psiw = ParetoSmooth.psis(logw; source="other")
            psis_ess = psiw.ess[1]
            pareto_k = psiw.pareto_k[1]
        catch
        end
        
        push!(results, (
            subj = subj,
            n_paths = n_paths,
            surrog_range = (minimum(loglik_surrog), maximum(loglik_surrog)),
            target_range = (minimum(loglik_target), maximum(loglik_target)),
            logw_range = (minimum(log_weights), maximum(log_weights)),
            logw_std = std(log_weights),
            simple_ess = simple_ess,
            psis_ess = psis_ess,
            pareto_k = pareto_k
        ))
    end
    
    return results
end

println("\n  Analyzing weights...")
results1 = sample_and_analyze(model_fit, surrogate, params_cur, tpm_book, hazmat_book, books, fbmats, absorbingstates)

for r in results1
    println("  Subject $(r.subj):")
    println("    Surrog LL: [$(round(r.surrog_range[1], digits=2)), $(round(r.surrog_range[2], digits=2))]")
    println("    Target LL: [$(round(r.target_range[1], digits=2)), $(round(r.target_range[2], digits=2))]")
    println("    Log-weight: [$(round(r.logw_range[1], digits=3)), $(round(r.logw_range[2], digits=3))], std=$(round(r.logw_std, digits=4))")
    println("    Simple ESS: $(round(r.simple_ess, digits=1)), PSIS ESS: $(round(r.psis_ess, digits=1)), Pareto-k: $(round(r.pareto_k, digits=3))")
end

# ============================================================================
# Test 2: Semi-Markov target (shape != 1)
# ============================================================================
println("\n--- Test 2: Weibull target (shape != 1) ---")
println("Expected: Markov surrogate should be worse proposal")

# Simulate with shape != 1
set_parameters!(model_sim, (
    h12 = [log(0.8), log(0.5)],  # shape=0.8 (decreasing hazard)
    h21 = [log(1.2), log(0.3)]   # shape=1.2 (increasing hazard)
))

sim_result2 = simulate(model_sim; paths=false, data=true, nsim=1)
simulated_data2 = sim_result2[1, 1]

trans2 = combine(groupby(simulated_data2, :id)) do gdf
    (n_trans = sum(gdf.statefrom .!= gdf.stateto),)
end
println("  Transitions per subject: ", trans2.n_trans)

model_fit2 = multistatemodel(h12, h21; data=simulated_data2, surrogate=:markov)
surrogate2 = model_fit2.markovsurrogate
surrogate_pars2 = get_log_scale_params(surrogate2.parameters)

# Rebuild TPM book for new model
for t in eachindex(books[1])
    compute_hazmat!(hazmat_book[t], surrogate_pars2, surrogate2.hazards, books[1][t], model_fit2.data)
    compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
end
fbmats2 = any(model_fit2.data.obstype .> 2) ? build_fbmats(model_fit2) : nothing

println("\n  Analyzing weights...")
params_cur2 = get_parameters_flat(model_fit2)
results2 = sample_and_analyze(model_fit2, surrogate2, params_cur2, tpm_book, hazmat_book, books, fbmats2, absorbingstates)

for r in results2
    println("  Subject $(r.subj):")
    println("    Surrog LL: [$(round(r.surrog_range[1], digits=2)), $(round(r.surrog_range[2], digits=2))]")
    println("    Target LL: [$(round(r.target_range[1], digits=2)), $(round(r.target_range[2], digits=2))]")
    println("    Log-weight: [$(round(r.logw_range[1], digits=3)), $(round(r.logw_range[2], digits=3))], std=$(round(r.logw_std, digits=4))")
    println("    Simple ESS: $(round(r.simple_ess, digits=1)), PSIS ESS: $(round(r.psis_ess, digits=1)), Pareto-k: $(round(r.pareto_k, digits=3))")
end

# ============================================================================
# Test 3: With TVC
# ============================================================================
println("\n--- Test 3: Model with TVC ---")
println("Expected: Should work similarly to without TVC")

# Build panel data with TVC
rows_tvc = []
for subj in 1:n_subj
    for t in 0.0:0.2:max_time-0.2
        x_val = t >= 2.0 ? 1.0 : 0.0  # Treatment starts at t=2
        push!(rows_tvc, (id=subj, tstart=t, tstop=t+0.2, statefrom=1, stateto=1, obstype=2, x=x_val))
    end
end
panel_data_tvc = DataFrame(rows_tvc)

h12_tvc = Hazard(@formula(0 ~ x), "wei", 1, 2)
h21_tvc = Hazard(@formula(0 ~ x), "wei", 2, 1)
model_sim_tvc = multistatemodel(h12_tvc, h21_tvc; data=panel_data_tvc, surrogate=:markov)

set_parameters!(model_sim_tvc, (
    h12 = [log(1.0), log(0.5), 0.3],  # shape=1, scale=0.5, beta=0.3
    h21 = [log(1.0), log(0.3), -0.2]  # shape=1, scale=0.3, beta=-0.2
))

sim_result_tvc = simulate(model_sim_tvc; paths=false, data=true, nsim=1)
simulated_data_tvc = sim_result_tvc[1, 1]

trans_tvc = combine(groupby(simulated_data_tvc, :id)) do gdf
    (n_trans = sum(gdf.statefrom .!= gdf.stateto),)
end
println("  Transitions per subject: ", trans_tvc.n_trans)

model_fit_tvc = multistatemodel(h12_tvc, h21_tvc; data=simulated_data_tvc, surrogate=:markov)
surrogate_tvc = model_fit_tvc.markovsurrogate
surrogate_pars_tvc = get_log_scale_params(surrogate_tvc.parameters)

# Rebuild infrastructure for TVC model
books_tvc = build_tpm_mapping(model_fit_tvc.data)
hazmat_book_tvc = build_hazmat_book(Float64, model_fit_tvc.tmat, books_tvc[1])
tpm_book_tvc = build_tpm_book(Float64, model_fit_tvc.tmat, books_tvc[1])

for t in eachindex(books_tvc[1])
    compute_hazmat!(hazmat_book_tvc[t], surrogate_pars_tvc, surrogate_tvc.hazards, books_tvc[1][t], model_fit_tvc.data)
    compute_tmat!(tpm_book_tvc[t], hazmat_book_tvc[t], books_tvc[1][t], cache)
end
fbmats_tvc = any(model_fit_tvc.data.obstype .> 2) ? build_fbmats(model_fit_tvc) : nothing

println("\n  Analyzing weights...")
params_cur_tvc = get_parameters_flat(model_fit_tvc)
results_tvc = sample_and_analyze(model_fit_tvc, surrogate_tvc, params_cur_tvc, tpm_book_tvc, hazmat_book_tvc, books_tvc, fbmats_tvc, absorbingstates)

for r in results_tvc
    println("  Subject $(r.subj):")
    println("    Surrog LL: [$(round(r.surrog_range[1], digits=2)), $(round(r.surrog_range[2], digits=2))]")
    println("    Target LL: [$(round(r.target_range[1], digits=2)), $(round(r.target_range[2], digits=2))]")
    println("    Log-weight: [$(round(r.logw_range[1], digits=3)), $(round(r.logw_range[2], digits=3))], std=$(round(r.logw_std, digits=4))")
    println("    Simple ESS: $(round(r.simple_ess, digits=1)), PSIS ESS: $(round(r.psis_ess, digits=1)), Pareto-k: $(round(r.pareto_k, digits=3))")
end

println("\n" * "=" ^ 70)
println("Summary")
println("=" ^ 70)
println("Test 1 (exponential): Should have very low log-weight variance")
println("Test 2 (semi-Markov): May have higher log-weight variance")
println("Test 3 (with TVC): Should behave similarly to Test 1 if shape=1")
println("\nIf log-weight std >> 0, proposal is inefficient and MCEM may need many paths")
println("=" ^ 70)
