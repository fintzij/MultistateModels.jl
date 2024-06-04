# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

dat = 
    DataFrame(id = ones(5),
              tstart = collect(0.0:0.2:0.8),
              tstop = collect(0.2:0.2:1.0),
              statefrom = [1,2,1,2,1],
              stateto = [2,1,2,1,3],
              obstype = ones(5),
              x = repeat(rand(1), 5))

# create multistate model object with the simulated data
meshsize = 100000
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; monotonic = "nonmonotonic", degree = 3, knots = [0.25, 0.5, 0.75], meshsize = meshsize) # healthy -> ill
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; monotonic = "increasing", degree = 3, knots = [0.25, 0.5, 0.75], meshsize = meshsize) # healthy -> dead
h14 = Hazard(@formula(0 ~ 1), "sp", 1, 4; monotonic = "decreasing", degree = 3, knots = [0.25, 0.5, 0.75], meshsize = meshsize)
h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; monotonic = "nonmonotonic", degree = 3, knots = [0.25, 0.5, 0.75], meshsize = meshsize) # healthy -> ill
h31 = Hazard(@formula(0 ~ 1 + x), "sp", 3, 1; monotonic = "increasing", degree = 3, knots = [0.25, 0.5, 0.75], meshsize = meshsize) # healthy -> dead
h41 = Hazard(@formula(0 ~ 1 + x), "sp", 4, 1; monotonic = "decreasing", degree = 3, knots = [0.25, 0.5, 0.75], meshsize = meshsize)

hazards = (h12, h13, h14, h21, h31, h41)
splinemod = multistatemodel(h12, h13, h14, h21, h31, h41; data = dat)

initialize_parameters!(splinemod; crude = true)