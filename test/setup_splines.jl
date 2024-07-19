# set up a MultistateModel object
using BSplineKit
using DataFrames
using Distributions
using MultistateModels

dat = 
    DataFrame(id = 1,
              tstart = 0.0,
              tstop = 1.0,
              statefrom = 1,
              stateto = 2,
              obstype = 1,
              x = rand(1))

# create multistate model object with the simulated data
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 3, knots = collect(0.2:0.2:0.8), extrapolation = "flat") 

h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 1, knots = [0.25, 0.5, 0.75]) 

h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; degree = 3, knots = collect(0.2:0.2:0.8)) 

h31 = Hazard(@formula(0 ~ 1 + x), "sp", 3, 1; degree = 1, knots = [0.25, 0.5, 0.75], extrapolation = "flat") 

h32 = Hazard(@formula(0 ~ 1), "sp", 3, 2; degree = 1, knots = [0.5, 0.8], extrapolation = "linear")

hazards = (h12, h13, h21, h31, h32)
splinemod = multistatemodel(h12, h13, h21, h31, h32; data = dat)

for h in eachindex(splinemod.hazards)
    copyto!(splinemod.parameters[h], rand(Normal(0,1), length(splinemod.parameters[h])))
end

# set the 1->3 parameters manually
splinemod.parameters[5] = log.([0.25, 0.7])
MultistateModels.remake_splines!(splinemod.hazards[2], splinemod.parameters[2])
MultistateModels.set_riskperiod!(splinemod.hazards[2])

# recombine parameters and compute risk periods
for h in eachindex(splinemod.hazards)
    MultistateModels.remake_splines!(splinemod.hazards[h], splinemod.parameters[h])
    MultistateModels.set_riskperiod!(splinemod.hazards[h])
end
