using DataFrames
using Distributions
using MultistateModels
using Random

# set up the very simplest model
nsubj = 100
ntimes = 10
dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
                tstart = repeat(0:(1/ntimes):(1 - 1/ntimes), outer = nsubj),
                tstop = repeat((1/ntimes):(1/ntimes):1, outer = nsubj),
                statefrom = fill(1, ntimes * nsubj),
                stateto = fill(2, ntimes * nsubj),
                obstype = fill(2, ntimes * nsubj))

h12e = Hazard(@formula(0 ~ 1), "exp", 1, 2)

mod = multistatemodel(h12e; data = dat)

# set_parameters!(mod, (h12 = (log(0.8), log(0.8)),))

# simulate
simdat = simulate(mod; paths = false, data = true)[1]

# set up model for inference
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0, knots = [0.0, 1.0], extrapolation = "flat")

model = multistatemodel(h12; data = simdat)
initialize_parameters!(model)

model_fitted = fit(model)
