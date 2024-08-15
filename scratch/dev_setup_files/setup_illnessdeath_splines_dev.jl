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

dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
              tstart = repeat(collect(0:(ntimes-1))/ntimes, outer = nsubj),
              tstop = repeat(collect(1:ntimes)/ntimes, outer  = nsubj),
              statefrom = ones(nsubj * ntimes),
              stateto = ones(nsubj * ntimes ),
              obstype = 2,
              trt = rand([0.0,1.0], nsubj * ntimes))

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1 + trt), "sp", 1, 2; degree = 1) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree = 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 2, knots = [0.4,]) # healthy -> dead
# h21 = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree = 0) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 3, knots = [0.2, 0.5, 0.8]) # ill -> dead

hazards = (h12, h13, h21, h23); 
splinemod = multistatemodel(h12, h13, h21, h23; data = dat)

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
crude_init!(modelsp)
fitted = fit(modelsp; compute_vcov = true, verbose = true)

using ArraysOfArrays, Optimization, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, RCall, StatsFuns, MacroTools, BSplineKit, RuntimeGeneratedFunctions, ParetoSmooth, LinearAlgebra, ArraysOfArrays, ForwardDiff, DiffResults, OptimizationOptimJL

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, _SplinePH, check_SamplingWeights, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs, extract_paths, MarkovSurrogate, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, get_subjinds, enumerate_hazards, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, check_data!, check_SamplingWeights, spline_hazards, check_CensoringPatterns, build_emat, _TotalHazardAbsorbing, build_fbmats, mcem_lml, mcem_lml_subj, remake_splines!, set_riskperiod!, set_crude_init!, init_par, _MarkovHazard, _SplineHazard, ComputeImportanceWeightsESS!, compute_loglik, spline_ests2coefs, spline_coefs2ests, init_par, remake_splines!, rectify_coefs!, make_optim_pars

optimize_surrogate = true; constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing; maxiter = 100; tol = 1e-3; α = 0.2; γ = 0.2; κ = 2.0; ess_target_initial = 50; max_ess = 10000; MaxSamplingEffort = 20; npaths_additional = 10; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = false; compute_vcov = true;
optim_pars = nothing; vcov_threshold = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; 
SamplingWeights = nothing;
