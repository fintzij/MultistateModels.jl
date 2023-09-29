# set up a MultistateModel object
using DataFrames
using MultistateModels

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2);
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3);
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
h23 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 3);

nsubj = 3

dat_exact = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [3, 3, 3],
              obstype = ones(3))

dat_exact2 = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [3, 3, 1],
              obstype = ones(3),
              trt = [0, 1, 0],
              age = [23, 32, 50])

dat_exact3 = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, 5*nsubj),
              stateto = fill(2, 5*nsubj),
              obstype = fill(1, 5*nsubj),
              trt = fill(1, 5*nsubj),
              age = fill(30, 5*nsubj))

dat_interval = 
    DataFrame(id = [1, 1, 2, 2, 3, 3],
              tstart = [0, 1, 0, 1, 0, 1],
              tstop = [1, 2, 1, 2, 1, 2],
              statefrom = [1, 1, 2, 2, 1, 1], # only the first statefrom counts for simulation
              stateto = [1, 1, 1, 1, 1, 1],
              obstype = ones(6),
              trt = [0, 0, 1, 1, 0, 0],
              age = [23, 23, 32, 32, 50, 50])

hazards = (h12, h21, h23, h13)
data = dat_exact2

# create multistate model object
msm_expwei = multistatemodel(h12, h23, h13, h21; data = dat_exact2)
msm_expwei2 = multistatemodel(h12, h23, h13, h21; data = dat_interval)
msm_expwei3 = multistatemodel(h12, h23, h13, h21; data = dat_exact3)

# simulate data for msm_expwei3 and put it in the model
simdat, paths = simulate(msm_expwei3; paths = true, data = true);
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2);
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3);
h21e = Hazard(@formula(0 ~ 1), "exp", 2, 1)
h23e = Hazard(@formula(0 ~ 1 + trt), "exp", 2, 3);
#data_sim3 = hcat(simdat[1], DataFrame(trt = fill(1, nrow(simdat[1])), age = fill(30, nrow(simdat[1]))))
data_sim3 = hcat(simdat[1][:,1:6], DataFrame(trt = sample([0, 1], nrow(simdat[1])), 
                                      age = sample(collect(20:70), nrow(simdat[1]))))

msm_expwei3 = multistatemodel(h12, h23e, h13, h21e; data=data_sim3)
# set model parameters
