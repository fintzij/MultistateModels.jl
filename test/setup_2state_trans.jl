# set up a MultistateModel object
using Chain
using DataFrames
using Distributions
using MultistateModels

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first

h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)

dat = 
    DataFrame(id = [1,1,1,2,2,2],
              tstart = [0, 10, 20, 0, 10, 20],
              tstop = [10, 20, 30, 10, 20, 30],
              statefrom = [1, 1, 1, 1, 1, 1],
              stateto = [2, 2, 1, 2, 1, 2],
              obstype = [1, 2, 3, 1, 0, 2],
              trt = [0, 1, 0, 1, 0, 1])

# create multistate model object
msm_2state_trans = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5, so log(1/5) = log(0.2). Hazard ratio of 1.3, so log(1.3)
set_parameters!(
    msm_2state_trans, 
    (h12 = [log(0.1), log(2)],
     h21 = [log(0.1), log(1), log(2)]))

# sample paths
path1 = 
    MultistateModels.SamplePath(
        1,
        [0.0, 8.2, 13.2, 30.0],
        [1, 2, 1, 1])

path2 = 
    MultistateModels.SamplePath(
        2,
        [0.0, 1.7, 27.2, 28.5, 29.3], 
        [1, 2, 1, 2, 1])

# # log-likelihood
# ll1 = MultistateModels.loglik(path[1], msm)
# ll2 = MultistateModels.loglik(path[2], msm)
