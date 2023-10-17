# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

nsubj = 100
ntimes = 5
dat = DataFrame(id = repeat(collect(1:100), inner = ntimes),
                tstart = repeat(collect(0:(10/ntimes):(10 - 10/ntimes)), outer = nsubj),
                tstop = repeat(collect((10/ntimes):(10/ntimes):10), outer = nsubj),
                statefrom = fill(1, nsubj * ntimes),
                stateto = fill(2, nsubj * ntimes),
                obstype = fill(2, nsubj * ntimes))

# append!(dat, DataFrame(id=1,tstart=10.0,tstop=20.0,statefrom=2,stateto=1,obstype=2))
# append!(dat, DataFrame(id=1,tstart=20.0,tstop=30.0,statefrom=2,stateto=1,obstype=3))
# sort!(dat, [:id,])

# create multistate model object
model = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5
set_parameters!(
    model, 
    (h12 = [log(0.4), log(0.8)],
     h21 = [log(0.4)]))

simdat, paths = simulate(model; paths = true, data = true);

# create multistate model object with the simulated data
model = multistatemodel(h12, h21; data = simdat[1])

MultistateModels.set_crude_init!(model)

# errors b/c there is no emat in the model
fitted = fit(model) 


# load libraries and functions
using ArraysOfArrays, Optimization, Optim, StatsModels, StatsFuns

constraints = nothing; nparticles = 10; maxiter = 100; tol = 1e-4; α = 0.1; γ = 0.05; κ = 1.5;
surrogate_parameter = nothing; ess_target_initial = 100; MaxSamplingEffort = 10;
verbose = true; return_ConvergenceRecords = true; return_ProposedPaths = true

using MultistateModels: build_tpm_mapping, MultistateMarkovModel, MultistateMarkovModelCensored, fit
