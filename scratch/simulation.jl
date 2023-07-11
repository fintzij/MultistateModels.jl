# set up a MultistateModel object
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

# observation window
tstart = 0.0
tstop = 3.0

# subject data
nsubj = 10000 # number of subject
dat = 
    DataFrame(id = collect(1:nsubj), # subject id
              tstart = fill(tstart, nsubj),
              tstop = fill(tstop, nsubj),
              statefrom = fill(1, nsubj), # state at time `tstart`
              stateto = fill(2, nsubj), # `stateto` is a placeholder before the simulation
              obstype = fill(1, nsubj)) # `obstype=1` indicates exactly observed data

# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
set_parameters!(
    model, 
    (h12 = [log(1.1)],
     h13 = [log(0.3)],
     h21 = [log(0.7)],
     h23 = [log(0.9)]))

#
# Simulate data
simdat, paths = simulate(model; paths = true, data = true)
# since `obstype=1`, `simdat` contains the exact paths
simdat[1]

#
# build model with simulated data
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

# fit model
# slow the first time because of compilation
# much faster the subsequent runs
model_fitted=fit(model)

summary_table, ll, AIC, BIC = MultistateModels.summary(model_fitted)
println(summary_table[:h12]) # MLE and 95% confidence intervals.