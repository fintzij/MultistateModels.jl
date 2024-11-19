# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)

nsubj = 50
ntimes = 3
step = 1.0
dat = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
              tstart = repeat(collect(range(0.0, step = step, length = ntimes)), outer = nsubj),
              tstop = repeat(collect(range(step, step = step, length = ntimes)), outer = nsubj),
              statefrom = fill(1, (ntimes*nsubj)),
              stateto = fill(2, (ntimes*nsubj)),
              obstype = fill(2, (ntimes*nsubj)))

# create multistate model object
mod = multistatemodel(h12, h21; data = dat)

# set model parameters
set_parameters!(mod, (h12 = [log(1.5), log(2)], h21 = [log(1.5), log(3)]))

# simulate dataset
dat = simulate(mod; data = true, paths = false, nsim = 1)[1]

# now fit it
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

mod_fit = multistatemodel(h12, h21; data = dat)

MultistateModels.initialize_parameters!(mod_fit)  
mod_fitted = fit(mod_fit)

# summarize
mod_fitted.vcov