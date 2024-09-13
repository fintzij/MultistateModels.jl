# set up a MultistateModel object
using DataFramesMeta
include("scratch/dev_setup_files/illnessdeath_sim/sim_funs.jl")

# set up model
# model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", ntimes = 12, nsubj = 10000)

ntimes = 12
nsubj = 10000
visitdays = ntimes == 12 ? [make_obstimes(12) for i in 1:nsubj] : 
                (ntimes == 6) ? [getindex(make_obstimes(12), [1, 3, 5, 7, 9, 11, 13]) for i in 1:nsubj] : [getindex(make_obstimes(12), [1, 4, 7, 10, 13]) for i in 1:nsubj]
data = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
            tstart = reduce(vcat, map(x -> x[Not(end)], visitdays)),
            tstop = reduce(vcat, map(x -> x[Not(1)], visitdays)),
            statefrom = fill(1, nsubj * ntimes),
            stateto = fill(1, nsubj * ntimes),
            obstype = fill(2, nsubj * ntimes))
model_sim = setup_model(;make_pars = false, data = data, family = "exp", ntimes = ntimes, nsubj = nsubj)
        
# simulate paths
data,paths = simulate(model_sim; nsim = 1, paths = true, data = true)
dat = reduce(vcat, map(x -> observe_subjdat(x, model_sim), paths))
# dat = data[1]

### set up model for fitting
model_fit = setup_model(; make_pars = false, data = dat, family = "exp", ntimes = ntimes)

# fit model
initialize_parameters!(model_fit)
model_fitted = fit(model_fit; verbose = true, compute_vcov = true, ess_target_initial = 50, α = 0.2, γ = 0.2, tol = 0.001) 

# set up for msm
@rlibrary msm

dat_msm = @chain dat begin
    groupby(:id)
    @combine(
        :id = [:id[1]; :id],
        :time = [:tstart[1]; :tstop],
        :state = [:statefrom[1]; :stateto])
end

qmat_msm = MultistateModels.calculate_crude(model_fit)
[]
diag(qmat_msm) .= 0


@rput(dat_msm)
@rput(qmat_msm)
fitted_msm = R"""
library(msm)
fitted_msm = msm(state ~ time, subject = id, data = dat_msm, qmatrix = qmat_msm, deathexact = 3)
fitted_msm
"""


# notes 
# - panel data matches
# - mixed panel/exact does not match but it seems to be an issue with msm