using DataFrames
using Distributions
using MultistateModels
# using Plots
using Random

Random.seed!(52787)

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

set_parameters!(model_sim, (h12 = (log(1.25), log(1)),))

# simulate
simdat = simulate(model_sim; paths = false, data = true)[1]

# set up model for inference
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2; degree = 3, knots = quantile(simdat.tstop[findall(simdat.stateto .== 2)], [0.25, 0.5, 0.75,]), extrapolation = "linear", monotone = 0)

model = multistatemodel(h12; data = simdat)

initialize_parameters!(model)
model_fitted = fit(model; tol = 0.001)

# plot(0:0.01:1, compute_hazard(0:0.01:1, model_fitted, :h12)); plot!(0:0.01:1, compute_hazard(0:0.01:1, model_sim, :h12))

println(model_fitted.parameters)
println(diag(model_fitted.vcov))