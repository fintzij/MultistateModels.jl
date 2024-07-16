
# get functions
include(pwd()*"/scratch/dev_setup_files/illnessdeath_sim/sim_funs.jl")

# run the simulation
simnum = 2098;
seed = 98;
family = 3; 
sims_per_subj = 20
nboot = 20
# work_function(;simnum = simnum, seed = seed, family = family, sims_per_subj = 1, nboot = 1)

# set up model for simulation
model_sim = setup_model(; make_pars = true, data = nothing, family = "gom", nsubj = 1000)
    
# simulate paths
paths = simulate(model_sim; nsim = 1, paths = true, data = false)
dat = reduce(vcat, map(x -> observe_subjdat(x, model_sim), paths))

### set up model for fitting
model= setup_model(; make_pars = false, data = dat, family = ["exp", "gom", "sp"][family])

# fit model
initialize_parameters!(model)
fitted = fit(model; verbose = true, compute_vcov = true) 

using ArraysOfArrays, Optimization, OptimizationOptimJL, StatsModels, ExponentialUtilities,  ArraysOfArrays, ElasticArrays, ForwardDiff, LinearAlgebra, StatsFuns, MacroTools, BSplineKit, RuntimeGeneratedFunctions, ParetoSmooth, LinearAlgebra, ArraysOfArrays, ForwardDiff, DiffResults


using MultistateModels: build_tpm_mapping, loglik, SMPanelData, build_hazmat_book, build_tpm_book, _TotalHazardTransient, SamplePath, sample_ecctmc, compute_hazmat!, compute_tmat!, sample_ecctmc!, _Spline, _SplinePH, check_SamplingWeights, draw_samplepath, mcem_mll, mcem_ase, loglik!, ExactData, SamplePath, get_subjinds, enumerate_hazards, create_tmat, check_data!, _Hazard, SplineHazard, build_hazards, survprob, call_haz, call_cumulhaz, total_cumulhaz, next_state_probs, extract_paths, MarkovSurrogate, extract_paths, get_subjinds, extract_sojourns, spline_hazards, check_SamplingWeights, parse_constraints, MPanelData, make_surrogate_model, DrawSamplePaths!, get_subjinds, enumerate_hazards, MarkovSurrogate, extract_paths, loglik, ExactData, ExactDataAD, check_data!, check_SamplingWeights, spline_hazards, check_CensoringPatterns, build_emat, _TotalHazardAbsorbing, build_fbmats, mcem_lml, mcem_lml_subj, recombine_parameters!, set_riskperiod!

nparticles = 10; maxiter = 150; tol = 1e-3; α = 0.05; γ = 0.05; κ = 4/3; verbose = true; surrogate = false; nsim = 1; subj = 1;
ess_target_initial = 100; MaxSamplingEffort = 20; npaths_additional = 25; verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

# SamplingWeights = nothing; 
CensoringPatterns = nothing; optimize_surrogate = true; SamplingWeights = nothing

constraints = nothing; surrogate_constraints = nothing; surrogate_parameters = nothing; compute_vcov = true
