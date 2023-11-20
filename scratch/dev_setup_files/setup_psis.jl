# set up a MultistateModel object
using ArraysOfArrays
using Chain
using DataFrames
using Distributions
using JLD2 # for saving files
using LinearAlgebra
using MultistateModels
using ParetoSmooth
using StatsBase
using Random

# import stuff
using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns, MacroTools, FunctionWrappers, RuntimeGeneratedFunctions

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs!, extract_paths, compute_spline_basis!, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, paretosmooth!

nparticles = 10; maxiter = 150; tol = 1e-8; α = 0.1; β = 0.3; γ = 0.05; κ = 3; verbose = true; surrogate = false; nsim = 1; data = true; paths = true; subj = 1;
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 25; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; optimize_surrogate = true

constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing

# load sim helpers
include(pwd()*"/scratch/dev_setup_files/regen_sim/sim_funs.jl");

# set up model
seed = 1; cens = 2; nulleff = 1; jobid = 20001; sims_per_subj = 10; nboot = 10

Random.seed!(seed)

# set up model for simulation
model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", n_per_arm = 750, nulleff = nulleff)
    
# simulate paths
paths = simulate(model_sim; nsim = 1, paths = true, data = false)[:,1]

# make dataset
dat = cens == 1 ? reduce(vcat, map(x -> observe_subjpath(x, model_sim), paths)) : 
        cens == 2 ? reduce(vcat, map(x -> observe_subjdat(x, model_sim; censor = false), paths)) : reduce(vcat, map(x -> observe_subjdat(x, model_sim; censor = true), paths))

### set up model for fitting
dat_collapsed, weights = collapse_data(dat)
model = setup_model(; make_pars = false, data = dat_collapsed, SamplingWeights = weights, family = "wei")

# try fitting
fitted = fit(model)