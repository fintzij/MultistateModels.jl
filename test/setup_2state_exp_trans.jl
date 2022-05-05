# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first

h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1 + trt), "exp", 2, 1)

dat = 
    DataFrame(id = [1,1,1,2,2,2],
              tstart = [0, 10, 20, 0, 10, 20],
              tstop = [10, 20, 100, 10, 20, 100],
              statefrom = [1, 1, 1, 1, 1, 1],
              stateto = [2, 2, 1, 2, 1, 2],
              obstype = [1, 1, 1, 1, 1, 1],
              trt = [0, 0, 0, 1, 1, 1])

# create multistate model object
msm = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5, so log(1/5) = log(0.2). Hazard ratio of 1.3, so log(1.3)
set_parameters!(
    msm, 
    (h12 = [log(0.2), log(2)],
     h21 = [log(0.1), log(0.5)]))