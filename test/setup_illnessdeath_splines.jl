# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # ill -> dead

nsubj = 200
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, 5*nsubj),
              stateto = fill(2, 5*nsubj),
              obstype = fill(2, 5*nsubj),
              trt = 1.0)


# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
# want mean time to event of 5
set_parameters!(
    model, 
    (h12 = [log(1.2), log(0.4)],
     h21 = [log(1.2), log(0.4)],
     h13 = [log(0.7), log(0.2)],
     h23 = [log(0.7), log(0.1)]))

simdat, paths = simulate(model; paths = true, data = true);

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1 + trt), "ms", 1, 2; degree = 3, df = 6) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "ms", 2, 1; degree = 2, df = 5) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "ms", 1, 3; degree = 1, df = 4) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "ms", 2, 3; degree = 0, df = 4) # ill -> dead

hazards = (h12, h13, h21, h23); data = simdat[1]
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall
using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz

nparticles = 10; maxiter = 100; tol = 1e-2; α = 0.1; β = 0.3; γ = 0.05; κ = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true