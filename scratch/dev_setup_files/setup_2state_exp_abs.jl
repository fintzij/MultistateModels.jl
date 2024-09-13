# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)

nsubj = 100
ntimes = 10
dat = DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, (5*nsubj)),
              stateto = fill(2, (5*nsubj)),
              obstype = fill(2, (5*nsubj)),
              trt = reduce(vcat, [sample([0,1], 1)[1] * ones(5) for i in 1:nsubj]))

# create multistate model object
msm = multistatemodel(h12; data = dat)

# set model parameters
set_parameters!(msm, (h12 = [log(0.2), log(2)],))

# weibull
h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)

dat = 
    DataFrame(id = [1,2],
              tstart = [0, 0],
              tstop = [100, 100],
              statefrom = [1, 1],
              stateto = [2, 2],
              obstype = [1, 1],
              trt = [0, 1])

# create multistate model object
msm_wei = multistatemodel(h12; data = dat)

# set model parameters
# want mean time to event of 5, so log(1/5) = log(0.2). Hazard ratio of 1.3, so log(1.3)
set_parameters!(msm_wei, (h12 = [log(1), log(0.2), log(2)],))