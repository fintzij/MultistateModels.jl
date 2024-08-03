
using DataFrames
using Distributions
using MultistateModels
using Random

# Random.seed!(0)

# set up the very simplest model
nsubj = 200
ntimes = 10
dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
                tstart = repeat(0:(1/ntimes):(1 - 1/ntimes), outer = nsubj),
                tstop = repeat((1/ntimes):(1/ntimes):1, outer = nsubj),
                statefrom = fill(1, ntimes * nsubj),
                stateto = fill(2, ntimes * nsubj),
                obstype = fill(2, ntimes * nsubj))

h12e = Hazard(@formula(0 ~ 1), "wei", 1, 2)

mod = multistatemodel(h12e; data = dat)

# set_parameters!(mod, (h12 = [log(0.8),],))
set_parameters!(mod, (h12 = (log(0.8), log(0.4)),))

# simulate
simdat = simulate(mod; paths = false, data = true)[1]

# set up model for inference
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 3, knots = quantile(simdat.tstop[findall(simdat.stateto .== 2)], [0.0, 0.25, 0.5, 0.75, 1.0]), extrapolation = "linear", monotone = -1)
# h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)

model = multistatemodel(h12; data = simdat)
initialize_parameters!(model)

model_fitted = fit(model, tol=1e-2)


plot(0:0.01:1, compute_hazard(0:0.01:1, model_fitted, :h12))
plot!(0:0.01:1, compute_hazard(0:0.01:1, mod, :h12))