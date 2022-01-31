# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2);
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3);
h23 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 3);
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)

dat_exact = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [3, 3, 3],
              obstype = zeros(3))

dat_exact2 = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [3, 3, 3],
              obstype = zeros(3),
              trt = [0, 1, 0],
              age = [23, 32, 50])

dat_interval = 
    DataFrame(id = [1, 1, 2, 2, 3, 3],
              tstart = [0, 1, 0, 1, 0, 1],
              tstop = [1, 2, 1, 2, 1, 2],
              statefrom = [1, 1, 2, 2, 1, 1], # only the first statefrom counts for simulation
              stateto = [1, 1, 1, 1, 1, 1],
              obstype = ones(6),
              trt = [0, 0, 1, 1, 0, 0],
              age = [23, 23, 32, 32, 50, 50])

# create multistate model object
msm_expwei = multistatemodel(h12, h23, h13, h21; data = dat_exact2)