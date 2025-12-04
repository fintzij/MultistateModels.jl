# =============================================================================
# Debug Script: Step through MCEM fitting line by line
# =============================================================================
# Run this script in the Julia REPL to set up everything, then you can manually
# step through the fit() function code and inspect variables.
#
# Usage:
#   1. Start Julia REPL: julia --project=.
#   2. include("scratch/debug_stepthrough.jl")
#   3. All variables will be set up - you can now manually run fit() code
# =============================================================================

using MultistateModels
using DataFrames
using Random
using LinearAlgebra
using Statistics

# Import internal functions we need to access
import MultistateModels: get_parameters_flat, set_parameters!, get_log_scale_params,
    nest_params, build_tpm_mapping, build_hazmat_book, build_tpm_book,
    compute_hazmat!, compute_tmat!, build_fbmats, DrawSamplePaths!,
    ComputeImportanceWeightsESS!, loglik, loglik!, SMPanelData,
    mcem_mll, mcem_ase, compute_loglik, set_surrogate!,
    fit_phasetype_surrogate, ProposalConfig, resolve_proposal_config,
    MarkovSurrogate, build_phasetype_tpm_book, build_fbmats_phasetype,
    build_phasetype_emat_expanded, compute_phasetype_marginal_loglik,
    compute_markov_marginal_loglik

using ExponentialUtilities
using ElasticArrays
using Optimization
using OptimizationOptimJL
using Distributions

println("=" ^ 70)
println("DEBUG SETUP: Reversible Model with TVC (failing case)")
println("=" ^ 70)

# Set seed for reproducibility
Random.seed!(52787)

# =============================================================================
# 1. CREATE DATA
# =============================================================================
println("\n1. Creating data...")

# More intervals per subject (e.g. 100 intervals of length 0.05 over [0,5])
nsubj = 2000
intervals_per_subj = 100
tstarts = collect(0:0.05:4.95)   # 100 start times
tstops  = collect(0.05:0.05:5.0) # 100 stop times

statefrom = fill(1, nsubj * intervals_per_subj)
subj_x1 = repeat([0, 1], nsubj ÷ 2)          # alternating covariate per subject
datTVC = DataFrame(
    id = repeat(1:nsubj, inner=intervals_per_subj),
    tstart = repeat(tstarts, nsubj),
    tstop = repeat(tstops, nsubj),
    statefrom = statefrom,
    stateto = fill(1, nsubj * intervals_per_subj),
    obstype = fill(1, nsubj * intervals_per_subj),
    x1 = repeat(subj_x1, inner=intervals_per_subj)
)

# Create simulation model
model_sim = multistatemodel(
    Hazard(@formula(0 ~ 1 + x1), "wei", 1, 2),
    Hazard(@formula(0 ~ 1 + x1), "wei", 2, 1);
    data=datTVC, surrogate=:markov
)

# True parameters
true_h12 = [log(1.0), log(0.5), 0.5]   # shape=1, scale=0.5, x1_coef=0.5
true_h21 = [log(1.0), log(0.3), -0.2]  # shape=1, scale=0.3, x1_coef=-0.2
set_parameters!(model_sim, (h12=true_h12, h21=true_h21))

# Simulate data
simdat,paths = simulate(model_sim; paths=true, data=true)
println("   Simulated $(nrow(simdat[1])) observations for $(length(unique(simdat[1].id))) subjects")

# matrix of exact data
exactdat = MultistateModels.paths_to_dataset(paths[1])  # Get first simulation dataset

# tack on the time varying x1 covariate (from simdat) to exactdat, need to use searchsortedlast to align
exactdat.x1 = similar(exactdat.tstart)
for i in 1:nrow(exactdat)
    subj = exactdat.id[i]
    tstart = exactdat.tstart[i]
    # find matching row in simdat
    row_idx = searchsortedlast(simdat[1].tstart[simdat[1].id .== subj], tstart)
    exactdat.x1[i] = simdat[1].x1[simdat[1].id .== subj][row_idx]
end


# =============================================================================
# 2. CREATE FITTING MODEL
# =============================================================================
println("\n2. Creating fitting model...")

model = multistatemodel(
    Hazard(@formula(0 ~ 1 + x1), "wei", 1, 2),
    Hazard(@formula(0 ~ 1 + x1), "wei", 2, 1);
    data=exactdat, surrogate=:markov
)

fitted = fit(model)
get_parameters_natural(fitted)


println("   Model type: $(typeof(model))")
println("   Initial params: $(get_parameters_flat(model))")
println("   Surrogate params: $(model.markovsurrogate.parameters.flat)")

# =============================================================================
# 3. SET UP FIT() PARAMETERS (copy from modelfitting.jl)
# =============================================================================
println("\n3. Setting up fit() parameters...")

# These are the default arguments to fit()
proposal = :auto
constraints = nothing
solver = nothing
maxiter = 100
tol = 1e-2
ascent_threshold = 0.1
stopping_threshold = 0.1
ess_increase = 2.0
ess_target_initial = 50
max_ess = 10000
max_sampling_effort = 20
npaths_additional = 10
block_hessian_speedup = 2.0
acceleration = :none
verbose = true
return_convergence_records = true
return_proposed_paths = false
compute_vcov = false

# Derived settings
use_squarem = acceleration === :squarem
proposal_config = resolve_proposal_config(proposal, model)
use_phasetype = proposal_config.type === :phasetype

println("   use_phasetype: $use_phasetype")
println("   proposal_config: $proposal_config")

# =============================================================================
# 4. MCEM INITIALIZATION (lines ~560-620 of modelfitting.jl)
# =============================================================================
println("\n4. MCEM initialization...")

keep_going = true
iter = 0
is_converged = false

nsubj = length(model.subjectindices)
println("   nsubj: $nsubj")

absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))
println("   absorbingstates: $absorbingstates")

params_cur = get_parameters_flat(model)
println("   params_cur (initial): $params_cur")

ess_target = ess_target_initial
ess_cur = zeros(nsubj)
psis_pareto_k = zeros(nsubj)

# Initialize containers
samplepaths = [sizehint!(Vector{MultistateModels.SamplePath}(), ess_target_initial * max_sampling_effort * 20) for i in 1:nsubj]
loglik_surrog = [sizehint!(Vector{Float64}(undef, 0), ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]
loglik_target_cur = [sizehint!(Vector{Float64}(undef, 0), ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]
loglik_target_prop = [sizehint!(Vector{Float64}(undef, 0), ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]
_logImportanceWeights = [sizehint!(Vector{Float64}(undef, 0), ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]
ImportanceWeights = [sizehint!(Vector{Float64}(undef, 0), ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

# Build fbmats if needed
if any(model.data.obstype .> 2)
    fbmats = build_fbmats(model)
else
    fbmats = nothing
end

# Traces
mll_trace = Vector{Float64}()
ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0)
parameters_trace = ElasticArray{Float64, 2}(undef, length(params_cur), 0)

# =============================================================================
# 5. SURROGATE SETUP (lines ~623-670 of modelfitting.jl)
# =============================================================================
println("\n5. Surrogate setup...")

# Check if surrogate exists
if isnothing(model.markovsurrogate)
    error("MCEM requires a Markov surrogate")
end

# CHECK: Does surrogate need fitting?
surrog_pars = model.markovsurrogate.parameters.flat
println("   Surrogate params before check: $surrog_pars")
println("   Are all zeros? $(all(isapprox.(surrog_pars, 0.0; atol=1e-6)))")

if all(isapprox.(surrog_pars, 0.0; atol=1e-6))
    println("   >>> Surrogate has default params, fitting via MLE...")
    set_surrogate!(model; type=:markov, method=:mle, verbose=true)
    println("   >>> Surrogate params after MLE: $(model.markovsurrogate.parameters.flat)")
end

markov_surrogate = model.markovsurrogate

# Build phase-type if needed
phasetype_surrogate = nothing
tpm_book_ph = nothing
hazmat_book_ph = nothing
fbmats_ph = nothing
emat_ph = nothing

if use_phasetype
    println("   Building phase-type surrogate...")
    phasetype_surrogate = fit_phasetype_surrogate(model, markov_surrogate; 
                                                   config=proposal_config, verbose=true)
end

surrogate = markov_surrogate

# =============================================================================
# 6. TPM SETUP (lines ~650-680 of modelfitting.jl)
# =============================================================================
println("\n6. TPM setup...")

books = build_tpm_mapping(model.data)
hazmat_book_surrogate = build_hazmat_book(Float64, model.tmat, books[1])
tpm_book_surrogate = build_tpm_book(Float64, model.tmat, books[1])
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_surrogate[1]), ExpMethodGeneric())

# Compute TPMs
surrogate_pars = get_log_scale_params(surrogate.parameters)
println("   surrogate_pars for TPM: $([collect(p) for p in surrogate_pars])")

for t in eachindex(books[1])
    compute_hazmat!(hazmat_book_surrogate[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
    compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
end

# Build phase-type TPMs if needed
if use_phasetype
    tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
    fbmats_ph = build_fbmats_phasetype(model, phasetype_surrogate)
    emat_ph = build_phasetype_emat_expanded(model, phasetype_surrogate)
end

# Compute normalizing constant
if use_phasetype
    NormConstantProposal = compute_phasetype_marginal_loglik(model, phasetype_surrogate, emat_ph)
else
    NormConstantProposal = compute_markov_marginal_loglik(model, markov_surrogate)
end
println("   NormConstantProposal: $NormConstantProposal")

# =============================================================================
# 7. INITIAL PATH SAMPLING (lines ~682-715 of modelfitting.jl)
# =============================================================================
println("\n7. Initial path sampling...")
println("   Drawing sample paths until ESS target is reached...")

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
    tpm_book_surrogate = tpm_book_surrogate, 
    hazmat_book_surrogate = hazmat_book_surrogate, 
    books = books, 
    npaths_additional = npaths_additional, 
    params_cur = params_cur, 
    surrogate = surrogate, 
    psis_pareto_k = psis_pareto_k,
    fbmats = fbmats,
    absorbingstates = absorbingstates,
    phasetype_surrogate = phasetype_surrogate,
    tpm_book_ph = tpm_book_ph,
    hazmat_book_ph = hazmat_book_ph,
    fbmats_ph = fbmats_ph,
    emat_ph = emat_ph)

println("   Paths per subject: min=$(minimum(length.(samplepaths))), max=$(maximum(length.(samplepaths)))")
println("   ESS: min=$(round(minimum(ess_cur), digits=1)), max=$(round(maximum(ess_cur), digits=1))")

# =============================================================================
# 8. INSPECT INITIAL STATE
# =============================================================================
println("\n8. Initial state inspection...")

# Check first few subjects
for i in 1:3
    npaths = length(samplepaths[i])
    println("\n   Subject $i: $npaths paths")
    println("   ESS: $(round(ess_cur[i], digits=2))")
    println("   Pareto-k: $(round(psis_pareto_k[i], digits=3))")
    
    if npaths > 0
        # Sample of log-likelihoods
        println("   First 3 paths:")
        for j in 1:min(3, npaths)
            ll_surrog = loglik_surrog[i][j]
            ll_target = loglik_target_cur[i][j]
            log_w = _logImportanceWeights[i][j]
            w = ImportanceWeights[i][j]
            println("     Path $j: ll_surrog=$(round(ll_surrog, digits=2)), ll_target=$(round(ll_target, digits=2)), log_w=$(round(log_w, digits=2)), w=$(round(w, digits=4))")
        end
    end
end

# Compute initial MLL
mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
println("\n   Initial MLL (Q): $(round(mll_cur, digits=3))")

# =============================================================================
# READY FOR MANUAL STEPPING
# =============================================================================
println("\n" * "=" ^ 70)
println("SETUP COMPLETE - You can now manually step through the MCEM loop")
println("=" ^ 70)
println("""
Available variables:
  - model: the fitting model
  - params_cur: current parameters (flat, log-scale)
  - samplepaths: sample paths for each subject
  - loglik_surrog: surrogate log-likelihoods
  - loglik_target_cur: target log-likelihoods (current params)
  - loglik_target_prop: target log-likelihoods (proposed params)
  - ImportanceWeights: normalized importance weights
  - _logImportanceWeights: log unnormalized importance weights
  - ess_cur: effective sample size per subject
  - mll_cur: current marginal log-likelihood
  - surrogate / markov_surrogate: Markov surrogate
  - phasetype_surrogate: phase-type surrogate (if use_phasetype)

To run one M-step manually:
  # Set up optimization
  optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
  prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights))
  
  # Solve
  params_prop_optim = solve(remake(prob, u0=Vector(params_cur), p=SMPanelData(model, samplepaths, ImportanceWeights)), Optim.LBFGS())
  params_prop = params_prop_optim.u
  
  # Compute new log-likelihoods
  loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
  
  # Compute new MLL
  mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
  
  # Compare
  println("MLL change: \$(mll_prop - mll_cur)")
""")
