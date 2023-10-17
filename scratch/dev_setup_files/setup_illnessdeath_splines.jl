# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

dat = 
    DataFrame(id = ones(5),
              tstart = repeat(collect(0.0:2.0:8.0)),
              tstop = repeat(collect(2.0:2.0:10.0)),
              statefrom = [1,2,1,2,1],
              stateto = [2,1,2,1,3],
              obstype = ones(5),
              x = ones(Float64, 5))

              # create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 3) # healthy -> dead
h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; degree = 0) # ill -> healthy
h23 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 3; degree = 3) # ill -> dead

hazards = (h12, h13, h21, h23); 
splinemod = multistatemodel(h12, h13, h21, h23; data = dat)

set_parameters!(
    splinemod, 
    (h12 = [log(0.4)],
     h13 = [log(0.4), 0.1, 0.1, 0.1],
     h21 = [log(0.4), 0.1],
     h23 = [log(0.4), 0.1, 0.1, 0.1, 0.1]))
