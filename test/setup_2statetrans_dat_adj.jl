# set up a MultistateModel object
using Chain
using DataFrames
using Distributions
using MultistateModels
using StatsBase

h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1 + trt), "exp", 2, 1)

dat = 
    DataFrame(id = repeat(collect(1:100), inner = 2),
              tstart = repeat([0.0,5.0], outer = 100),
              tstop = repeat([5.0,10.0], outer = 100),
              statefrom = fill(1, 200),
              stateto = fill(2, 200),
              obstype = fill(2, 200),
              trt = reduce(vcat, [sample([0,1], 2) for i in 1:100]))

# want different gap times
dat.tstop[1:2:199] = sample([4.0,5.0,6.0], 100)
dat.tstart[2:2:200] = dat.tstop[1:2:199]

# append!(dat, DataFrame(id=1,tstart=10.0,tstop=20.0,statefrom=2,stateto=1,obstype=2))
# append!(dat, DataFrame(id=1,tstart=20.0,tstop=30.0,statefrom=2,stateto=1,obstype=3))
# sort!(dat, [:id,])

# create multistate model object
msm_2state_transadj = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5 without treatment
# treatment speeds 1->2 and slows 2->1
set_parameters!(
    msm_2state_transadj, 
    (h12 = [log(0.2), log(1.5)],
     h21 = [log(0.2), log(2/3)]))

simdat, paths = simulate(msm_2state_transadj; paths = true, data = true);

# create multistate model object with the simulated data
msm_2state_transadj = multistatemodel(h12, h21; data = simdat[1])

set_parameters!(msm_2state_transadj, (h12 = randn(2), h21 = randn(2)))

model = msm_2state_transadj
using ArraysOfArrays, Optimization, OptimizationOptimJL, DifferentialEquations, ExponentialUtilities
using MultistateModels: build_tpm_mapping, loglik, PanelData, build_hazmat_book, build_tpm_book

