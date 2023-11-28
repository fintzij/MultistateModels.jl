# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

nsubj = 1000
ntimes = 10

dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
              tstart = repeat(collect(0:(ntimes-1))/ntimes, outer = nsubj),
              tstop = repeat(collect(1:ntimes)/ntimes, outer  = nsubj),
              statefrom = ones(nsubj * ntimes),
              stateto = ones(nsubj * ntimes ),
              obstype = 1)

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2; degree = 0) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3; degree = 0) # healthy -> dead
# h21 = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree = 0) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3; degree = 0) # ill -> dead

hazards = (h12, h13, h23); 
splinemod = multistatemodel(h12, h13, h23; data = dat)

set_parameters!(
    splinemod, 
    (h12 = [log(0.4)],
     h13 = [log(0.4)],
     h23 = [log(0.4)]))

simdat = simulate(splinemod; data = true, paths = false)[1]

# remake model
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 0) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 0) # ill -> dead

model = multistatemodel(h12, h13, h23; data = simdat)

# try to fit
set_crude_init!(model)
fitted = fit(model; compute_vcov = false)

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns, MacroTools, FunctionWrappers, RuntimeGeneratedFunctions

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs!, extract_paths, compute_spline_basis!, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, paretosmooth!

nparticles = 10; maxiter = 150; tol = 1e-8; α = 0.1; β = 0.3; γ = 0.05; κ = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true; subj = 1;
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 25; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; optimize_surrogate = true

constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing; compute_vcov = false



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
