# set up a MultistateModel object
using DataFrames
using MultistateModels
using StatsBase

# Stan will initialize parameters by sampling from N(0,1) unless given explicit parameters
# This isn't crazy because e.g. brms will center covariates first
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2);
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3);

nsubj = 30

dat = DataFrame(id = repeat(1:nsubj, inner = 10),
              tstart = repeat(0:9, outer = nsubj),
              tstop = repeat(1:10, outer = nsubj),
              statefrom = fill(1, nsubj * 10),
              stateto = fill(2, nsubj * 10),
              obstype = fill(2, 10 * nsubj))

hazards = (h12, h23)

# create multistate model object
msm_expwei = multistatemodel(h12, h23; data = dat)

# simulate data for msm_expwei3 and put it in the model
simdat, paths = simulate(msm_expwei; paths = true, data = true)

function getdat(i)
    inds = findall(simdat[1].id .== i)
    n = length(inds)
    DataFrame(id = fill(i, n),
            tstart = simdat[1].tstart[inds],
            tstop = [simdat[1].tstop[inds[Not(end)]]; paths[i].times[end]],
            statefrom = simdat[1].statefrom[inds],
            stateto = [simdat[1].stateto[inds[Not(end)]]; 3],
            obstype = [fill(2, n - 1); 1])
end

dat2 = reduce(vcat, [getdat(i) for i in 1:nsubj])
model_fit = multistatemodel(h12, h23; data = dat2)

# set model parameters
initialize_parameters!(model_fit)

# fit
model_fitted = fit(model_fit)
get_loglik(model_fitted)
estimate_loglik(model_fitted; min_ess = 1000, paretosmooth = false)