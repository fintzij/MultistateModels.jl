
using DataFrames
using Distributions
using MultistateModels

#
# Model setup for simulation

# healthy-ill-dead model
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

# parameters
par_true = (
    h12 = [log(1.1)],
    h13 = [log(0.3)],
    h21 = [log(0.7)],
    h23 = [log(0.9)])

# number of subjects
nsubj = 10000 # number of subject

# observation window
tstart = 0.0
tstop = 3.0

#
# Exactly observed data

# subject data
dat = 
    DataFrame(id        = collect(1:nsubj), # subject id
              tstart    = fill(tstart, nsubj),
              tstop     = fill(tstop, nsubj),
              statefrom = fill(1, nsubj), # state at time `tstart`
              stateto   = fill(2, nsubj), # `stateto` is a placeholder before the simulation
              obstype   = fill(1, nsubj)) # `obstype=1` indicates exactly observed data

# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
set_parameters!(model, par_true)

# Simulate data
simdat, paths = simulate(model; paths = true, data = true)
# since `obstype=1`, `simdat` contains the exact paths
println(simdat[1])

# build model with simulated data
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

# fit model
# slow the first time because of compilation, but much faster the subsequent runs
model_fitted = fit(model)

# summary of results
summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h12])
println(par_true[:h12]) # true value


#
# Panel data

# number of observations per subject
nobs_per_subj = 5
times_obs = collect(range(tstart, stop = tstop, length = nobs_per_subj+1))

# subject data
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = nobs_per_subj),
              tstart = repeat(times_obs[1:nobs_per_subj], outer = nsubj),
              tstop = repeat(times_obs[2:nobs_per_subj+1], outer = nsubj),
              statefrom = fill(1, nobs_per_subj*nsubj), # `statefrom` after `tstart=0` is a placeholder before the simulation
              stateto = fill(2, nobs_per_subj*nsubj), # `stateto` is a placeholder before the simulation
              obstype = fill(2, nobs_per_subj*nsubj)) # `obstype=2` indicates panel data

              
# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
set_parameters!(model, par_true)

# Simulate data
simdat, paths = simulate(model; paths = true, data = true)
# since `obstype=2`, `simdat` contains the panel observations
println(simdat[1][1:30,:])

# build model with simulated data
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])
set_parameters!(model, par_true)

# @Jason
# MultistateModels.set_crude_init!(model)

# fit model
# slow the first time because of compilation, but much faster the subsequent runs
model_fitted = fit(model)

# summary of results
summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h12])
println(par_true[:h12]) # true value

model_fitted.parameters

model.SamplingWeights
#check: i calculation of tpm, ii index of tpm