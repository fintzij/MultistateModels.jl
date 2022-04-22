# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first

h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)

dat = 
    DataFrame(id = [1,1,2,2],
              tstart = [0, 10, 0, 10],
              tstop = [10, 100, 10, 100],
              statefrom = [1, 1, 1, 1],
              stateto = [2, 2, 2, 2],
              obstype = zeros(4),
              trt = [0, 0, 1, 1])

# create multistate model object
msm = multistatemodel(h12; data = dat)

# set model parameters
# want mean time to event of 5, so log(1/5) = log(0.2). Hazard ratio of 1.3, so log(1.3)
set_parameters!(msm, (h12 = [log(0.2), log(2)],))