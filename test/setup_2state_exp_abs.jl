# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)

dat = 
    DataFrame(id = [1,1,1,2,2,2],
              tstart = [0, 10, 20, 0, 10, 20],
              tstop = [10, 20, 100, 10, 20, 100],
              statefrom = [1, 1, 1, 1, 1, 1],
              stateto = [2, 2, 2, 2, 2, 2],
              obstype = [1, 0, 1, 1, 0, 1],
              trt = [0, 0, 0, 1, 1, 1])

# create multistate model object
msm = multistatemodel(h12; data = dat)

# set model parameters
set_parameters!(msm, (h12 = [log(0.2), log(2)],))

# weibull
h12 = Hazard(@formula(0 ~ 1 + trt), "wei", 1, 2)

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