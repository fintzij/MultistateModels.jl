# set up a MultistateModel object
using DataFrames
using MultistateModels
using StatsBase

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first
h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2);
h13 = Hazard(@formula(0 ~ 1 + trt), "gom", 1, 3);

nsubj = 2
dat = DataFrame(id = collect(1:nsubj),
              tstart = zeros(nsubj),
              tstop = fill(10.0, nsubj),
              statefrom = ones(Int64, nsubj),
              stateto =  ones(Int64, nsubj),
              obstype =  ones(Int64, nsubj),
              trt = [0, 1])

# create multistate model object
msm_gom = multistatemodel(h12, h13; data = dat)
