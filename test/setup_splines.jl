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
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 3, knots = collect(0.2:0.2:0.8), add_boundaries = false, extrapolation = "flat") 

h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 1, knots = [0.25, 0.5, 0.75], add_boundaries = true) 

h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; degree = 3, knots = collect(0.2:0.2:0.8), add_boundaries = false) 

h31 = Hazard(@formula(0 ~ 1 + x), "sp", 3, 1; degree = 1, knots = [0.25, 0.5, 0.75], add_boundaries = true, extrapolation = "flat") 

hazards = (h12, h13, h21, h31)
splinemod = multistatemodel(h12, h13, h21, h31; data = dat)

initialize_parameters!(splinemod; crude = true)