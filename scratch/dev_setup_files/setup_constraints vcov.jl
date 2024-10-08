# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # ill -> dead

nsubj = 100
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 10),
              tstart = repeat(collect(0.0:1.0:9.0), outer = nsubj),
              tstop = repeat(collect(1.0:1.0:10.0), outer = nsubj),
              statefrom = fill(1, 10*nsubj),
              stateto = fill(2, 10*nsubj),
              obstype = fill(1, 10*nsubj))

# create multistate model object
model = multistatemodel(h12, h13, h23; data = dat)

# set model parameters
set_parameters!(
    model, 
    (h12 = [log(1.3), log(0.4)],
     h13 = [log(1.3), log(0.2)],
     h23 = [log(1.3), log(0.1)]))

simdat, paths = simulate(model; paths = true, data = true);

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # ill -> dead

hazards = (h12, h13, h23); data = simdat[1]
datc, weights = collapse_data(simdat[1])
model = multistatemodel(h12, h13, h23; data = datc, SamplingWeights = weights)

set_parameters!(
    model, 
    (h12 = [0.1, log(0.5)],
     h13 = [0.1, log(0.5)],
     h23 = [0.1, log(0.3)]))

constraints = make_constraints(
    cons = [:(h12_shape - h13_shape), :(h12_shape - h23_shape)], 
    lcons = [0.0, 0.0],
    ucons = [0.0, 0.0])
hazards = model.hazards

surrogate_constraints = make_constraints(cons = [:(h12_x - h13_x),], lcons = [0.0, ], ucons = [0.0,])

fitted = fit(model)

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns, MacroTools, FunctionWrappers, RuntimeGeneratedFunctions

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs, extract_paths, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, _Hazard, extract_paths

nparticles = 10; maxiter = 150; tol = 1e-8; ascent_threshold = 0.1; β = 0.3; stopping_threshold = 0.05; ess_increase = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true; subj = 1
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 10; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; optimize_surrogate = true

 constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing