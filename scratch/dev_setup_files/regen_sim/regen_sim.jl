using ArraysOfArrays
using Base.Threads
using Chain
using DataFrames
using Distributions
using LinearAlgebra
using MultistateModels
using StatsBase
using Random

# get command line arguments
simnum = 4989; seed = 989; cens = 3; nulleff = 1; sims_per_subj = 20; nboot = 1000

# get functions
include(pwd()*"/scratch/dev_setup_files/regen_sim/sim_funs.jl");

# run the simulation
# jobtime = @elapsed results = work_function(;simnum = simnum, seed = seed, cens = cens, nulleff = nulleff, sims_per_subj = sims_per_subj, nboot = nboot)



using ArraysOfArrays, Optimization, OptimizationOptimJL, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra,  RCall, StatsFuns, MacroTools, RuntimeGeneratedFunctions, ParetoSmooth

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs, extract_paths, MarkovSurrogate, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, get_subjinds, enumerate_hazards, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, check_data!, check_SamplingWeights, mcem_mll, build_fbmats

optimize_surrogate = true; constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing;  maxiter = 200; tol = 1e-3; α = 0.05; γ = 0.05; κ = 4/3; ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 10; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = false; compute_vcov = true


