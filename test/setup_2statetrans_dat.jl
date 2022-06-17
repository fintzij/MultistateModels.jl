# set up a MultistateModel object
using Chain
using DataFrames
using Distributions
using MultistateModels

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

dat = 
    DataFrame(id = collect(1:100),
              tstart = fill(0.0, 100),
              tstop = fill(10.0, 100),
              statefrom = fill(1, 100),
              stateto = fill(2, 100),
              obstype = fill(1, 100))

# create multistate model object
msm_2state_trans = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5, so log(1/5) = log(0.2). Hazard ratio of 1.3, so log(1.3)
set_parameters!(
    msm_2state_trans, 
    (h12 = [log(0.2)],
     h21 = [log(0.5)]))

paths = simulate(msm_2state_trans; paths = true, data = true);
