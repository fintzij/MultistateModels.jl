# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

nsubj = 100
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, 5*nsubj),
              stateto = fill(2, 5*nsubj),
              obstype = fill(2, 5*nsubj))

# dat = DataFrame(id = collect(1:nsubj),
#             tstart = fill(0.0, nsubj),
#             tstop = fill(10.0, nsubj),
#             statefrom = fill(1, nsubj),
#             stateto = fill(2, nsubj),
#             obstype = fill(1, nsubj))

# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
# set_parameters!(
#     model, 
#     (h12 = [log(1.3), log(0.4)],
#      h21 = [log(1.3), log(0.4)],
#      h13 = [log(0.7), log(0.2)],
#      h23 = [log(0.7), log(0.1)]))
set_parameters!(
    model, 
    (h12 = [log(0.4)],
     h21 = [log(0.4)],
     h13 = [log(0.2)],
     h23 = [log(0.1)]))

simdat, paths = simulate(model; paths = true, data = true);

simdat[1][!,:x] = randn(size(simdat[1], 1))

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 3) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # ill -> dead

hazards = (h12, h13, h21, h23); data = simdat[1]
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

set_parameters!(
    model, 
    (h12 = [log(0.5), 0.1],
     h13 = [log(0.5), 0.1],
     h21 = [0.0, log(0.3)],
     h23 = [0.0, log(0.3)]))

constraints = make_constraints(cons = [:(h12_x - h13_x), ], lcons = [0.0,], ucons = [0.0,])
hazards = model.hazards

surrogate_constraints = make_constraints(cons = [:(h12_x - h13_x),], lcons = [0.0, ], ucons = [0.0,])

surrogate_parameters = 
    (h12 = [log(0.23), 0.0],
    h13 = [log(0.58), 0.0],
    h21 = [log(0.1)],
    h23 = [log(0.5)])

fitted = fit(model; constraints = constraints, surrogate_constraints = surrogate_constraints, surrogate_parameters = surrogate_parameters)

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns, MacroTools, FunctionWrappers, RuntimeGeneratedFunctions

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs!, extract_paths, compute_spline_basis!, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, MarkovSurrogate

nparticles = 10; maxiter = 100; tol = 1e-2; α = 0.1; β = 0.3; γ = 0.05; κ = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true; subj = 1
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 10; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true
SamplingWeights = nothing; CensoringPatterns = nothing; optimize_surrogate = true
