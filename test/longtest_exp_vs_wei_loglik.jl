using Chain
using DataFrames
using DataFramesMeta
using Distributions
using MultistateModels
using MultistateModels: loglik

# helpers
function make_obstimes()    
    times = collect(0:0.25:1) .+ vcat([0.0; (rand(Beta(5,5), 4) * 0.2 .- 0.1)])    
    return times
end

# set up dataset
nsubj = 500
ntimes = 4
visitdays = [make_obstimes() for i in 1:nsubj]
data = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
            tstart = reduce(vcat, map(x -> x[Not(end)], visitdays)),
            tstop = reduce(vcat, map(x -> x[Not(1)], visitdays)),
            statefrom = fill(1, nsubj * ntimes),
            stateto = fill(1, nsubj * ntimes),
            obstype = fill(2, nsubj * ntimes))

# transition intensities
h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)

# models
model_sim = multistatemodel(h12_exp; data = data)
set_parameters!(model_sim, (h12 = [log(1.5)],))

# simulate data
dat_raw, paths = simulate(model_sim; paths = true, data = true)

# remake data to use full panel data
model_exp = multistatemodel(h12_exp; data = dat_raw[1])

constraints = make_constraints(
    cons = [:(h12_shape;), ],
    lcons = [0.0,],
    ucons =[0.0,])
model_wei = multistatemodel(h12_wei; data = dat_raw[1])


# fit 
fit_exp = fit(model_exp)
fit_wei = fit(model_wei; constraints = constraints, ess_target_initial = 10)

paths_exp = simulate(fit_exp; paths = true, data = false, nsim = 100)
paths_wei = simulate(fit_wei; paths = true, data = false, nsim = 100)

# compute log like
subj_ll_exp = zeros(size(paths_exp, 2))
subj_ll_wei = zeros(size(paths_wei, 2))

for k in 1:size(paths_exp, 2)
    for j in 1:size(paths_exp, 1)
        subj_ll_exp[k] += loglik(fit_exp.parameters, paths_exp[j,k], fit_exp.hazards, fit_exp)
        subj_ll_wei[k] += loglik(fit_wei.parameters, paths_exp[j,k], fit_wei.hazards, fit_wei)
    end
end

# pretty close
mean(subj_ll_exp)
mean(subj_ll_wei)


# sanity check - matches exponential, not weibull via mcem
obj = 0.0
dat1 = dat_raw[1]
scale = 1 / exp(fit_exp.parameters[1][1])
for k in 1:nrow(dat_raw[1])
    obj += dat1.stateto[k] == 1 ? logccdf(Exponential(scale), dat1.tstop[k] - dat1.tstart[k]) : logcdf(Exponential(scale), dat1.tstop[k] - dat1.tstart[k])
end