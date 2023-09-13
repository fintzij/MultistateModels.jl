# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

nsubj = 10000
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, 5*nsubj),
              stateto = fill(2, 5*nsubj),
              obstype = fill(1, 5*nsubj))

dat = DataFrame(id = collect(1:nsubj),
            tstart = fill(0.0, nsubj),
            tstop = fill(10.0, nsubj),
            statefrom = fill(1, nsubj),
            stateto = fill(2, nsubj),
            obstype = fill(1, nsubj))

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
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 0) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree = 0) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 0) # ill -> dead

hazards = (h12, h13, h21, h23); data = simdat[1]
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

set_parameters!(
    model, 
    (h12 = [log(0.4)] + randn(1),
     h13 = [log(0.2)] + randn(1),
     h21 = [log(0.4)] + randn(1),
     h23 = [log(0.1)] + randn(1)))

# set_parameters!(
#     model, 
#     (h12 = [log(1.3), log(0.4)] .+ randn(2),
#      h21 = [log(1.3), log(0.4)] .+ randn(2),
#      h13 = [log(0.7), log(0.2)] .+ randn(2),
#      h23 = [log(0.7), log(0.1)] .+ randn(2)))

fitted = fit(model)

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns
RCall.@rlibrary splines2
using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs!, extract_paths, compute_spline_basis!, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights

# nparticles = 10; maxiter = 100; tol = 1e-2; α = 0.1; β = 0.3; γ = 0.05; κ = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true; subj = 1
SamplingWeights = nothing; CensoringPatterns = nothing
