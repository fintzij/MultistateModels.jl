# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

function summarize_paths(paths)
    pfs = mean(map(x -> all(x.states .== 1), paths))
    prog = mean(map(x -> 2 ∈ x.states, paths))
    die_wprog = mean(map(x -> all([2,3] .∈ Ref(x.states)), paths))
    die_noprog = mean(map(x -> (3 ∈ x.states) & !(2 ∈ x.states), paths))
    ests = (pfs = pfs, 
            prog = prog,
            die_wprog = die_wprog, 
            die_noprog = die_noprog)
    return ests
end

nsubj = 100
ntimes = 10

data = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
              tstart = repeat(collect(0:(ntimes-1))/ntimes, outer = nsubj),
              tstop = repeat(collect(1:ntimes)/ntimes, outer  = nsubj),
              statefrom = ones(nsubj * ntimes),
              stateto = ones(nsubj * ntimes ),
              obstype = 2)

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 1, monotonic = "decreasing") # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 0) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 0) # ill -> dead

hazards = (h12, h13, h23); 
splinemod = multistatemodel(h12, h13, h23; data = data)

set_parameters!(
    splinemod, 
    (h12 = [log(0.4)],
     h13 = [log(0.4)],
     h23 = [log(0.4)]))

simdat, paths = simulate(splinemod; data = true, paths = true)
summarize_paths(paths)
# 1 - cdf(Exponential(1/0.8), 1)
# 0.2844 / (1 - 0.4431)

# note - hazards and cumulative hazards are probably fine. simulated incidence of progression + death match expected when simulating w/ spline w/degree 0

# log likelihood 0 deg sp and exp haz is the same
# 

# remake model
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 0) # healthy -> dead
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0) # healthy -> ill
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 0) # ill -> dead

modelsp = multistatemodel(h12, h13, h23; data = simdat[1])
model = deepcopy(modelsp)

h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3; degree = 0) # healthy -> dead
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2; degree = 0) # healthy -> ill
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3; degree = 0) # ill -> dead

modelexp = multistatemodel(h12, h13, h23; data = simdat[1])

# try to fit
set_crude_init!(modelsp)
fitted = fit(modelsp; compute_vcov = true, verbose = true)

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns, MacroTools, FunctionWrappers, RuntimeGeneratedFunctions, ParetoSmooth

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _MSpline, _SplineHazard, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs!, extract_paths, MarkovSurrogate, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, get_subjinds, enumerate_hazards, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, check_data!, check_SamplingWeights

nparticles = 10; maxiter = 150; tol = 1e-8; α = 0.1; β = 0.3; γ = 0.05; κ = 4/3; verbose = true; surrogate = false; nsim = 1; subj = 1;
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 25; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; optimize_surrogate = true; SamplingWeights = nothing

constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing; compute_vcov = true


