using DataFrames
using Distributions
using MultistateModels
using Plots
using Random

# Random.seed!(0)

# set up the very simplest model
nsubj = 100
ntimes = 10
dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
                tstart = repeat(0:(1/ntimes):(1 - 1/ntimes), outer = nsubj),
                tstop = repeat((1/ntimes):(1/ntimes):1, outer = nsubj),
                statefrom = fill(1, ntimes * nsubj),
                stateto = fill(2, ntimes * nsubj),
                obstype = fill(2, ntimes * nsubj))

h12e = Hazard(@formula(0 ~ 1), "wei", 1, 2)

model_sim = multistatemodel(h12e; data = dat)

# set_parameters!(mod, (h12 = [log(0.8),],))
set_parameters!(model_sim, (h12 = (log(1.25), log(1.5)),))

# simulate
simdat = simulate(model_sim; paths = false, data = true)[1]

# set up model for inference
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 3, knots = [0.0; quantile(simdat.tstop[findall(simdat.stateto .== 2)], [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]); 1.0], extrapolation = "linear", monotone = 1)
# h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0, boundaryknots = [0.0, 1.0])

model = multistatemodel(h12; data = simdat)
initialize_parameters!(model)

model_fitted = fit(model; tol = 0.01, γ = 0.01)

plot(0:0.01:1, compute_hazard(0:0.01:1, model_fitted, :h12)); plot!(0:0.01:1, compute_hazard(0:0.01:1, model_sim, :h12))