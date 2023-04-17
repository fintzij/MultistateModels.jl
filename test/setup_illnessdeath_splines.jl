# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # healthy -> ill
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1) # ill -> healthy
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3) # healthy -> dead
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # ill -> dead

nsubj = 200
dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, 5*nsubj),
              stateto = fill(2, 5*nsubj),
              obstype = fill(2, 5*nsubj))


# create multistate model object
model = multistatemodel(h12, h13, h21, h23; data = dat)

# set model parameters
# want mean time to event of 5
set_parameters!(
    model, 
    (h12 = [log(1.2), log(0.4)],
     h21 = [log(1.2), log(0.4)],
     h13 = [log(0.7), log(0.2)],
     h23 = [log(0.7), log(0.1)]))

simdat, paths = simulate(model; paths = true, data = true);

# create multistate model object with the simulated data
msm_2state_trans = multistatemodel(h12, h21; data = simdat[1])