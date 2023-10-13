# set up a MultistateModel object
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

dat = 
    DataFrame(id = repeat(collect(1:100), inner = 2),
              tstart = repeat([0.0,5.0], outer = 100),
              tstop = repeat([5.0,10.0], outer = 100),
              statefrom = fill(1, 200),
              stateto = fill(2, 200),
              obstype = fill(2, 200))


# append!(dat, DataFrame(id=1,tstart=10.0,tstop=20.0,statefrom=2,stateto=1,obstype=2))
# append!(dat, DataFrame(id=1,tstart=20.0,tstop=30.0,statefrom=2,stateto=1,obstype=3))
# sort!(dat, [:id,])

# create multistate model object
msm_2state_trans = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5
set_parameters!(
    msm_2state_trans, 
    (h12 = [log(0.2)],
     h21 = [log(0.2)]))

simdat, paths = simulate(msm_2state_trans; paths = true, data = true);

# create multistate model object with the simulated data
msm_2state_trans = multistatemodel(h12, h21; data = simdat[1])