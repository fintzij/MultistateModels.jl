using ArraysOfArrays
using Chain
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using MultistateModels
using RCall
using StatsBase
using Random


# get functions
include(pwd()*"/scratch/dev_setup_files/illnessdeath_sim/sim_funs.jl");

# run the simulation
simnum = 1; seed = 1; family = 2; sims_per_subj = 20; nboot = 10
work_function(;simnum = simnum, seed = seed, family = family, sims_per_subj = 1, nboot = 1)

# save results
# # CSV.write("/data/fintzijr/multistate/sim1_illnessdeath/illnessdeath_results_$simnum.csv", results)
# CSV.write("/data/liangcj/illnessdeath/illnessdeath_results_$simnum.csv", results)


# meshrange = [0.0, 2.3]
# meshsize = 100
# mesh = collect(LinRange(meshrange[1], meshrange[2], meshsize))

# lb = 2.29998
# ub = 2.29999

# lind = Int64(ceil((lb - meshrange[1]) / meshrange[2] * meshsize))

# uind = Int64(ceil((ub - meshrange[1]) / meshrange[2] * meshsize))

# lind = lind == 0 ? 1 : lind == meshsize ? meshsize - 1 : lind
# uind = uind == lind ? lind + 1 : uind


using ArraysOfArrays, Optimization, OptimizationOptimJL, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, OptimizationOptimisers, RCall, Plots, StatsFuns, MacroTools, FunctionWrappers, RuntimeGeneratedFunctions, ParetoSmooth, LinearAlgebra

RCall.@rlibrary splines2

using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _MSpline, _ISplineIncreasing, _ISplineDecreasing, _MSplinePH, _ISplineIncreasingPH, _ISplineDecreasingPH, check_SamplingWeights, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs, extract_paths, MarkovSurrogate, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, get_subjinds, enumerate_hazards, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, check_data!, check_SamplingWeights, spline_hazards, check_CensoringPatterns, build_emat, _TotalHazardAbsorbing, build_fbmats

nparticles = 10; maxiter = 150; tol = 1e-3; α = 0.05; γ = 0.05; κ = 4/3; verbose = true; surrogate = false; nsim = 1; subj = 1;
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 25; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; optimize_surrogate = true; SamplingWeights = nothing

constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing; compute_vcov = true


