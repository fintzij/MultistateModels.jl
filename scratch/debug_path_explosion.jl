"""
Debug script to investigate path explosion in MCEM.
This script runs a single MCEM iteration and breaks out early to inspect
the importance weights and ESS when paths start to explode.
"""

using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

# Import internal functions
import MultistateModels: 
    Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_log_scale_params, SamplePath, 
    nest_params, build_tpm_mapping, build_hazmat_book, build_tpm_book,
    compute_hazmat!, compute_tmat!, build_fbmats, loglik,
    DrawSamplePaths!, ComputeImportanceWeightsESS!

using ExponentialUtilities

println("=" ^ 70)
println("DEBUG: Path Explosion Investigation")
println("=" ^ 70)

Random.seed!(12345)

# True parameters (exponential = Weibull with shape=1)
true_params = (
    h12 = [log(1.0), log(0.3)],  # log(shape), log(scale)
    h21 = [log(1.0), log(0.25)]
)

# Build panel data
n_subj = 30
rows = []
for subj in 1:n_subj
    for (tstart, tstop) in [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)]
        push!(rows, (id=subj, tstart=tstart, tstop=tstop, statefrom=1, stateto=1, obstype=2))
    end
end
panel_data = DataFrame(rows)

# Create and set up model
model_sim = multistatemodel(
    Hazard(@formula(0 ~ 1), "wei", 1, 2),
    Hazard(@formula(0 ~ 1), "wei", 2, 1);
    data=panel_data, surrogate=:markov
)
set_parameters!(model_sim, true_params)

# Simulate
println("\n--- Simulating data ---")
sim_result = simulate(model_sim; paths=true, data=true, nsim=1)
simulated_data = reduce(vcat, sim_result[1][:, 1])
simulated_paths = sim_result[2][:, 1]

# Count transitions
function count_transitions(paths)
    n12, n21 = 0, 0
    for path in paths
        for i in 1:(length(path.states)-1)
            path.states[i] == 1 && path.states[i+1] == 2 && (n12 += 1)
            path.states[i] == 2 && path.states[i+1] == 1 && (n21 += 1)
        end
    end
    return n12, n21
end
n12, n21 = count_transitions(simulated_paths)
println("Transitions: 1→2=$n12, 2→1=$n21")

# Create panel model for fitting
model = multistatemodel(
    Hazard(@formula(0 ~ 1), "wei", 1, 2),
    Hazard(@formula(0 ~ 1), "wei", 2, 1);
    data=simulated_data, surrogate=:markov
)

# Get target and surrogate params
nsubj = length(model.subjectindices)
params_cur = get_parameters_flat(model)  # Initial target params
surrogate = model.markovsurrogate
surrogate_pars = get_log_scale_params(surrogate.parameters)

println("\n--- Initial parameters ---")
println("Target params: $params_cur")
println("Surrogate params: $(surrogate.parameters.flat)")

# Build TPM infrastructure
books = build_tpm_mapping(model.data)
hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
tpm_book = build_tpm_book(Float64, model.tmat, books[1])
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

for t in eachindex(books[1])
    compute_hazmat!(hazmat_book[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
    compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
end

# Build fbmats
fbmats = build_fbmats(model)
absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

# Initialize containers
ess_target = 25.0
ess_cur = zeros(nsubj)
psis_pareto_k = zeros(nsubj)

samplepaths = [Vector{SamplePath}() for _ in 1:nsubj]
loglik_surrog = [Vector{Float64}() for _ in 1:nsubj]
loglik_target_cur = [Vector{Float64}() for _ in 1:nsubj]
loglik_target_prop = [Vector{Float64}() for _ in 1:nsubj]
_logImportanceWeights = [Vector{Float64}() for _ in 1:nsubj]
ImportanceWeights = [Vector{Float64}() for _ in 1:nsubj]

println("\n--- Drawing initial sample paths ---")

# Draw initial paths manually to inspect
max_paths_per_subj = 100  # Cap for debugging

for i in 1:nsubj
    subj_inds = model.subjectindices[i]
    
    # Draw paths until ESS target or max paths
    while ess_cur[i] < ess_target && length(samplepaths[i]) < max_paths_per_subj
        # Draw a single path
        path = MultistateModels.draw_samplepath(i, model, tpm_book, hazmat_book, books[2], fbmats, absorbingstates)
        push!(samplepaths[i], path)
        
        # Compute likelihoods
        ll_surrog = loglik(surrogate_pars, path, surrogate.hazards, model)
        target_pars = nest_params(params_cur, model.parameters)
        ll_target = loglik(target_pars, path, model.hazards, model)
        
        push!(loglik_surrog[i], ll_surrog)
        push!(loglik_target_cur[i], ll_target)
        push!(loglik_target_prop[i], 0.0)
        push!(_logImportanceWeights[i], ll_target - ll_surrog)
        push!(ImportanceWeights[i], 1.0)
        
        # Recompute ESS
        if length(_logImportanceWeights[i]) > 1
            log_w = _logImportanceWeights[i]
            max_log_w = maximum(log_w)
            w = exp.(log_w .- max_log_w)
            w_normalized = w ./ sum(w)
            ess_cur[i] = 1 / sum(w_normalized .^ 2)
        else
            ess_cur[i] = 1.0
        end
    end
end

println("\n--- Path counts and ESS ---")
for i in 1:nsubj
    n_paths = length(samplepaths[i])
    println("Subject $i: $n_paths paths, ESS=$(round(ess_cur[i], digits=2))")
end

println("\n--- Importance weight diagnostics ---")
for i in 1:min(5, nsubj)
    log_w = _logImportanceWeights[i]
    if length(log_w) > 0
        println("\nSubject $i:")
        println("  n_paths: $(length(log_w))")
        println("  log_w range: [$(round(minimum(log_w), digits=2)), $(round(maximum(log_w), digits=2))]")
        println("  log_w mean: $(round(mean(log_w), digits=2))")
        println("  log_w std: $(round(std(log_w), digits=2))")
        println("  ESS: $(round(ess_cur[i], digits=2))")
        
        # Check individual components
        println("  Sample log-lik target: $(round.(loglik_target_cur[i][1:min(3,end)], digits=2))")
        println("  Sample log-lik surrog: $(round.(loglik_surrog[i][1:min(3,end)], digits=2))")
    end
end

println("\n--- Check if target and surrogate are very different ---")
# At true params, are the likelihoods similar?
set_parameters!(model, true_params)
params_true = get_parameters_flat(model)

for i in 1:min(3, nsubj)
    if length(samplepaths[i]) > 0
        path = samplepaths[i][1]
        ll_surrog = loglik(surrogate_pars, path, surrogate.hazards, model)
        target_pars_true = nest_params(params_true, model.parameters)
        ll_target_true = loglik(target_pars_true, path, model.hazards, model)
        println("Subject $i, path 1: ll_target=$ll_target_true, ll_surrog=$ll_surrog, diff=$(ll_target_true - ll_surrog)")
    end
end

println("\n" * "=" ^ 70)
println("Debug complete")
println("=" ^ 70)
