# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels
using StatsBase

h12 = Hazard(@formula(0 ~ 1 + trt), "gom", 1, 2)
h21 = Hazard(@formula(0 ~ 1 + trt), "gom", 2, 1)

nsubj = Int64(200)

dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = 5),
              tstart = repeat(collect(0.0:2.0:8.0), outer = nsubj),
              tstop = repeat(collect(2.0:2.0:10.0), outer = nsubj),
              statefrom = fill(1, (5*nsubj)),
              stateto = fill(2, (5*nsubj)),
              obstype = fill(2, (5*nsubj)),
              trt = reduce(vcat, [sample([0,1], 1)[1] * ones(5) for i in 1:nsubj]))

# create multistate model object
msm_2state_transadj = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5 without treatment
# treatment speeds 1->2 and slows 2->1
set_parameters!(
    msm_2state_transadj, 
    (h12 = [log(10), log(0.01), 0.5],
     h21 = [log(10), log(0.01), 0.5]))

simdat, paths = simulate(msm_2state_transadj; paths = true, data = true);

# create multistate model object with the simulated data
model = multistatemodel(h12, h21; data = simdat[1])

constraints = make_constraints(
    cons = [:(h12_trt - h21_trt),], 
    lcons = [0.0,],
    ucons = [0.0,])

surrogate_constraints = deepcopy(constraints)
surrogate_parameters = (h12 = [0.0, 0.0], h21 = [0.0, 0.0])

initialize_parameters!(model; surrogate_constraints = surrogate_constraints, surrogate_parameters = surrogate_parameters, constraints = constraints)

using MultistateModels: calculate_crude, fit_surrogate, init_par, set_parameters!, make_surrogate_model, initialize_parameters!, _ExponentialPH, _WeibullPH, _GompertzPH, _MSplinePH, _ISplineIncreasingPH, _ISplineDecreasingPH, set_crude_init!

constraints = nothing; surrogate_constraints = nothing; crude = false; surrogate_parameters = nothing