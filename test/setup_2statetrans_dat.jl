# set up a MultistateModel object
using Chain
using DataFrames
using Distributions
using MultistateModels

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

dat = 
    DataFrame(id = collect(1:100),
              tstart = fill(0.0, 100),
              tstop = fill(10.0, 100),
              statefrom = fill(1, 100),
              stateto = fill(2, 100),
              obstype = fill(1, 100))


append!(dat, DataFrame(id=1,tstart=10.0,tstop=20.0,statefrom=2,stateto=1,obstype=2))
append!(dat, DataFrame(id=1,tstart=20.0,tstop=30.0,statefrom=2,stateto=1,obstype=3))
sort!(dat, [:id,])

# create multistate model object
msm_2state_trans = multistatemodel(h12, h21; data = dat)

# set model parameters
# want mean time to event of 5, so log(1/5) = log(0.2). Hazard ratio of 1.3, so log(1.3)
set_parameters!(
    msm_2state_trans, 
    (h12 = [log(0.2)],
     h21 = [log(0.5)]))

simdat, paths = simulate(msm_2state_trans; paths = true, data = true);

# create multistate model object with the simulated data
model = multistatemodel(h12, h21; data = simdat[1])