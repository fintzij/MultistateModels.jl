# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

#
# Model setup for simulation
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3) # ill -> dead

nsubj = 10
dat = 
    DataFrame(id = collect(1:nsubj),
              tstart = fill(0.0, nsubj),
              tstop = fill(3.0, nsubj),
              statefrom = fill(1, nsubj),
              stateto = fill(2, nsubj),
              obstype = fill(1, nsubj))

# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
set_parameters!(
    model, 
    (h12 = [log(1.1)],
     h21 = [log(0.7)],
     h13 = [log(0.2)],
     h23 = [log(0.8)]))

#
# Simulate data
simdat, paths = simulate(model; paths = true, data = true)
# since `obstype=1`, `simdat` contains the exact paths
simdat[1]

#
# build model with simulated data
model = multistatemodel(h12, h13, h21, h23; data = simdat[1])

model_fitted=fit(model)