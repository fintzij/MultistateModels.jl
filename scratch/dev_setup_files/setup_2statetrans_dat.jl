# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels
using Plots

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

nsubj = 100
ntimes = 10
dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
                tstart = repeat(collect(0:(10/ntimes):(10 - 10/ntimes)), outer = nsubj),
                tstop = repeat(collect((10/ntimes):(10/ntimes):10), outer = nsubj),
                statefrom = fill(1, nsubj * ntimes),
                stateto = fill(2, nsubj * ntimes),
                obstype = fill(2, nsubj * ntimes))


dat.tstop[Not(collect(10:ntimes:(ntimes*nsubj)))] .= dat.tstop[Not(collect(10:ntimes:(ntimes*nsubj)))] + rand(nsubj * (ntimes - 1))
dat.tstart[Not(collect(1:ntimes:(ntimes*nsubj)))] .= dat.tstop[Not(collect(10:ntimes:(ntimes*nsubj)))]

# create multistate model object
model = multistatemodel(h12, h23; data = dat);

# set model parameters
# want mean time to event of 5
set_parameters!(
    model, 
    (h12 = [log(1.5), log(0.4)],
     h23 = [log(1.5), log(0.4)]))

simdat, paths = simulate(model; paths = true, data = true);

# create multistate model object with the simulated data
model = multistatemodel(h12, h23; data = simdat[1]);

MultistateModels.set_crude_init!(model)

# fit model
fitted = fit(model; tol = 1e-4, ess_target_initial = 100);

# load libraries and functions
using ArraysOfArrays, Optimization, Optim, StatsModels, StatsFuns, ExponentialUtilities, ElasticArrays, BenchmarkTools, Profile, ProfileView, DiffResults, ForwardDiff

constraints = nothing; nparticles = 10; maxiter = 100; tol = 1e-2; ascent_threshold = 0.1; stopping_threshold = 0.05; ess_increase = 1.5;
surrogate_parameters = nothing; ess_target_initial = 10; MaxSamplingEffort = 10;
verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true; npaths_additional = 10

using MultistateModels: build_tpm_mapping, MultistateMarkovModel, MultistateMarkovModelCensored, fit, MarkovSurrogate, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!, SamplePath, fit_surrogate, DrawSamplePaths!, mcem_mll, loglik, SMPanelData, make_surrogate_model, loglik!, mcem_ase, draw_samplepath, ExactDataAD

# timing
# @btime solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights)), Optim.NewtonTrustRegion()) #

# # profile
# Profile.clear()
# ProfileView.@profview solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights)), Newton()) # hessian-based

# Profile.clear()
# ProfileView.@profview loglik(Vector(params_cur), SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights)) 



# marginal loglikelihood
plot(fitted.ConvergenceRecords.mll_trace, title="Marginal Logikelihood", label=nothing, linewidth=3)
    xlabel!("Monte Carlo EM iteration")
    ylabel!("Marginal loglikelihood")

# ess per subject and per iteration
plot(fitted.ConvergenceRecords.ess_trace', title="ESS per subject",legend = :outertopright, linewidth=3)
xlabel!("Monte Carlo EM iteration")
    ylabel!("ESS")

# trace of parameters
haznames = map(x -> String(fitted.hazards[x].hazname), collect(1:length(fitted.hazards)))
plot(fitted.ConvergenceRecords.parameters_trace', title="Trace of the parameters", linewidth=3) # label=permutedims(haznames),legend = :outertopright)
    xlabel!("Monte Carlo EM iteration")
    ylabel!("Parameters")
