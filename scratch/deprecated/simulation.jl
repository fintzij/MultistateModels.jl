
using DataFrames
#using Distributions
using MultistateModels

#
# Model setup for simulation

# healthy-ill-dead model
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

# parameters
par_true = (
    h12 = [log(1.1), log(1.5)],
    h13 = [log(0.3)],
    h21 = [log(0.7)],
    h23 = [log(0.9)])

# number of subjects
nsubj = 100 # number of subject

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
println(simdat[1][1:30,:])

# build model with simulated data
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

# fit model
# slow the first time because of compilation, but much faster the subsequent runs
model_fitted = fit(model)


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
              statefrom = 1, # `statefrom` after `tstart=0` is a placeholder before the simulation
              stateto   = 2, # `stateto` is a placeholder before the simulation
              obstype   = 2) # `obstype=2` indicates panel data

# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
set_parameters!(model, par_true)

# Simulate data
simdat, paths = simulate(model; paths = true, data = true)
# since `obstype=2`, `simdat` contains the panel observations
println(simdat[1][1:30,:])
# compare with the simulated paths
println(paths[1])

# build model with simulated data
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

# (optional) initialize the surrogate model
initialize_surrogate!(model)

# fit model
model_fitted = fit(model)


#
# Mixture of exactly observed and panel data

dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = nobs_per_subj),
              tstart = repeat(times_obs[1:nobs_per_subj], outer = nsubj),
              tstop = repeat(times_obs[2:nobs_per_subj+1], outer = nsubj),
              statefrom = fill(1, nobs_per_subj*nsubj),
              stateto = fill(2, nobs_per_subj*nsubj),
              obstype = repeat(collect(1:2), outer = Int64(nobs_per_subj*nsubj/2))) # mixture of `obstype=1` and `obstype=2`

model = multistatemodel(h12, h13, h21, h23; data = dat)
set_parameters!(model, par_true)
simdat, paths = simulate(model; paths = true, data = true)

# `simdat` contains panel observations (`obstype=2`) and intervals during which the paths are exactly observed (`obstype=1`)
println(simdat[1][1:30,:])
print(paths[1])

model = multistatemodel(h12, h13, h21, h23; data = simdat[1])
model_fitted = fit(model)

summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h12])
println(par_true[:h12]) # true value


#
# Covariates

# include a treatment effect for some transitions
h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

# parameters
par_true = (
    h12 = [log(1.1), log(1/3)], # multiplicative effect of treatmet is 1/3 (protective effect)
    h13 = [log(0.3)],
    h21 = [log(0.7)],
    h23 = [log(0.9)])

# subject data with a binary covariate (trt)
dat =
  DataFrame(id        = collect(1:nsubj), # subject id
            tstart    = fill(tstart, nsubj),
            tstop     = fill(tstop, nsubj),
            statefrom = fill(1, nsubj), # state at time `tstart`
            stateto   = fill(2, nsubj), # `stateto` is a placeholder before the simulation
            obstype   = fill(1, nsubj), # `obstype=1` indicates exactly observed data
            trt       = repeat(collect(0:1), outer = Int64(nsubj/2))) # odd subjects receive the control (trt=1) and even subjects receive the treatment (trt=1)

model = multistatemodel(h12, h13, h21, h23; data = dat)
simdat, paths = simulate(model; paths = true, data = true)

par_init = (
    h12 = [log(1.1), log(1/3)] .+ randn(2), # multiplicative effect of treatmet is 1/3 (protective effect)
    h13 = [log(0.3) + randn(1)[1]],
    h21 = [log(0.7) + randn(1)[1]],
    h23 = [log(0.9) + randn(1)[1]])


model = multistatemodel(h12, h13, h21, h23; data = simdat[1])
set_parameters!(model, par_init)

model_fitted = fit(model)
summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h21])
println(par_true[:h21]) # true value





#
# Semi-Marov model with panel data

# semi-Markov model
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

# parameters
par_true = (
    h12 = [log(1.5), log(1)],
    h13 = [log(0.3)],
    h21 = [log(0.7)],
    h23 = [log(0.9)])

# subject data
nsubj=10000
tstart = 0.0
tstop = 3.0
nobs_per_subj = 10
times_obs = collect(range(tstart, stop = tstop, length = nobs_per_subj+1))
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = nobs_per_subj),
              tstart = repeat(times_obs[1:nobs_per_subj], outer = nsubj),
              tstop = repeat(times_obs[2:nobs_per_subj+1], outer = nsubj),
              statefrom = fill(1, nobs_per_subj*nsubj), # `statefrom` after `tstart=0` is a placeholder before the simulation
              stateto = fill(2, nobs_per_subj*nsubj), # `stateto` is a placeholder before the simulation
              obstype = fill(2, nobs_per_subj*nsubj)) # `obstype=2` indicates panel data

# simulate artificial data
model = multistatemodel(h12, h13, h21, h23; data = dat)
set_parameters!(model, par_true)
simdat, paths = simulate(model; paths = true, data = true)

# set up semi-Markov model
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

# get mle from Markov process
h12_markov = Hazard(@formula(0 ~ 1), "exp", 1, 2)
model_markov = multistatemodel(h12_markov, h13, h21, h23; data = simdat[1])
model_markov_fitted = fit(model_markov)
mle_markov=model_markov_fitted.parameters

# use mle from Markov model
par_mle_markov = (
    h12 = [log(1), mle_markov[1][1]],
    h13 = mle_markov[2],
    h21 = mle_markov[3],
    h23 = mle_markov[4])
set_parameters!(model,par_mle_markov)
#for i in model.hazards
#    set_par_to = init_par(i, log(crude_par[i.statefrom, i.stateto]))
#    set_parameters!(model, NamedTuple{(i.hazname,)}((set_par_to,)))
#end



model_fitted = fit(model; verbose=true)
summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h12])
println(par_true[:h12])

# add noise to true parameters
sd=0.2
par_noise = (
    h12 = [log(1.5), log(1)] .+ rand(Normal(0,sd), 2),
    h13 = [log(0.3)] .+ rand(Normal(0,sd), 1),
    h21 = [log(0.7)] .+ rand(Normal(0,sd), 1),
    h23 = [log(0.9)] .+ rand(Normal(0,sd), 1))
set_parameters!(model,par_noise)

model_fitted = fit(model; verbose=true, nparticles = 100)
summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h12])
println(par_true[:h12])