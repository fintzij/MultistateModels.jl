using DataFrames
using Distributions
using MultistateModels

# set up the very simplest model
nsubj = 1000
dat = DataFrame(id = collect(1:nsubj),
                tstart = zeros(nsubj),
                tstop = ones(nsubj),
                statefrom = ones(Int64, nsubj),
                stateto = ones(Int64, nsubj) .+ 1,
                obstype = ones(Int64, nsubj))

h12e = Hazard(@formula(0 ~ 1), "exp", 1, 2)

mod = multistatemodel(h12e; data = dat)

# simulate
simdat = simulate(mod; paths = false, data = true)[1]

# set up model for inference
h12s = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0, knots = [0.2, 0.8], add_boundaries = false)

model = multistatemodel(h12s; data = simdat)
initialize_parameters!(model)

model_fitted = fit(model)

# notes
# degree 0 with no extrapolation checks out
# degree 0 with extrapolation is incorrect