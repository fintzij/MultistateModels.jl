# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # ill -> dead

nsubj = 1000
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, 5*nsubj),
              stateto = fill(2, 5*nsubj),
              obstype = fill(2, 5*nsubj))

nsubj=2

dat =
    DataFrame(id = repeat(collect(1:nsubj), inner=3),
              tstart = [[1], [2], [2,3], [1], [2], [2,3,4]],
              tstop = [[2], [2,3], [1], [2], [2,3,4], [1]]
    )


tmp = DataFrame(tstop = [[2], [2,3], [1], [4]])
    
# censor some times and set other intervals to exactly observed
dat.obstype[findall(dat.tstop .== 8.0)] .= 0
dat.obstype[findall(dat.tstart .== 8.0)] .= 0

# emission matrix


# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
# want mean time to event of 5
set_parameters!(
    model, 
    (h12 = [log(1.3), log(0.4)],
     h21 = [log(1.3), log(0.4)],
     h13 = [log(0.7), log(0.2)],
     h23 = [log(0.7), log(0.1)]))

simdat, paths = simulate(model; paths = true, data = true);
 
set_parameters!(
    model, 
    (h12 = [log(1.3), log(0.4)] .+ randn(2),
     h21 = [log(1.3), log(0.4)] .+ randn(2),
     h13 = [log(0.7), log(0.2)] .+ randn(2),
     h23 = [log(0.7), log(0.1)] .+ randn(2)))

fitted = fit(model)

using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots
using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs!, extract_paths, compute_spline_basis!, extract_paths, get_subjinds, extract_sojourns, spline_hazards

# nparticles = 10; maxiter = 100; tol = 1e-2; ascent_threshold = 0.1; β = 0.3; stopping_threshold = 0.05; ess_increase = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true; subj = 1
