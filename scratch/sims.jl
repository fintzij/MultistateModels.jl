using CSV
using DataFrames
using Distributions
using MultistateModels

# First fit a Weibull model to the ACTT-1 data to get reasonable parameters to simulate from
actt = CSV.read("/home/liangcj/scratch/actt1_mm.csv", DataFrame)

# rename states 1 and 2 as state 3 then subtract 2 from all states so states start at 1
actt.stateto[actt.stateto .< 4] .= 3
actt.statefrom .= actt.statefrom .- 2
actt.stateto .= actt.stateto .- 2

h21 = Hazard(@formula(0~1), "exp", 2, 1) # hosp to recovered
h23 = Hazard(@formula(0~1), "exp", 2, 3) # hosp to suppl O2
h32 = Hazard(@formula(0~1), "exp", 3, 2) # suppl O2 to hosp
h34 = Hazard(@formula(0~1), "exp", 3, 4) # suppl O2 to NIPPV/high flow
h35 = Hazard(@formula(0~1), "exp", 3, 5) # suppl O2 to vent
h36 = Hazard(@formula(0~1), "exp", 3, 6) # suppl O2 to death
h43 = Hazard(@formula(0~1), "exp", 4, 3) # NIPPV/high flow to suppl O2
h45 = Hazard(@formula(0~1), "exp", 4, 5) # NIPPV/high flow to vent
h46 = Hazard(@formula(0~1), "exp", 4, 6) # NIPPV/high flow to death
h54 = Hazard(@formula(0~1), "exp", 5, 4) # vent to NIPPV/high flow
h56 = Hazard(@formula(0~1), "exp", 5, 6) # vent to death

model = multistatemodel(h21, h23, 
                        h32, h34, h35, h36, 
                        h43, h45, h46, 
                        h54, h56; 
                        data = actt)



# specify hazards for allowable transitions
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2) # hosp, no icu -> icu
h14 = Hazard(@formula(0 ~ 1), "wei", 1, 4) # hosp, no icu -> recovered
h15 = Hazard(@formula(0 ~ 1), "wei", 1, 4) # hosp, no icu -> dead
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3) # icu -> hosp, yes icu
h25 = Hazard(@formula(0 ~ 1), "wei", 2, 5) # icu -> dead
h34 = Hazard(@formula(0 ~ 1), "wei", 3, 4) # hosp, yes icu -> recovered
h35 = Hazard(@formula(0 ~ 1), "wei", 3, 5) # hosp, yes icu -> dead

# parameters
par_true = (
    h12 = [log(1.5), log(1.5)],
    h14 = [log(1.5), log(1.5)],
    h15 = [log(1.5), log(1.5)],
    h23 = [log(1.5), log(1.5)],
    h25 = [log(1.5), log(1.5)],
    h34 = [log(1.5), log(1.5)],
    h35 = [log(1.5), log(1.5)]
)

# subject data
nsubj = 1000
# 200-250 per arm because we are only using OS5 subjects
# ACTT-1 collapse
# Hosp no ICU (1): 4-5 if they've never been to 6 or 7
# In ICU: 6-7
# Hosp w/ ICU: 4-5 if they've been to 6 or 7
# Recover: 1-3
# Dead: 8
# tstart/tstop: study_day
# obstype: 2 for everything

tstart = 0.0
tstop = 3.0
nobs_per_subj = 5
times_obs = collect(range(tstart, stop = tstop, length = nobs_per_subj+1))

dat = 
    DataFrame(id = repeat(collect(1:nsubj), inner = nobs_per_subj),
              tstart = repeat(times_obs[1:nobs_per_subj], outer = nsubj),
              tstop = repeat(times_obs[2:nobs_per_subj+1], outer = nsubj),
              statefrom = fill(1, nobs_per_subj*nsubj), # `statefrom` after `tstart=0` is a placeholder before the simulation
              stateto = fill(2, nobs_per_subj*nsubj), # `stateto` is a placeholder before the simulation
              obstype = fill(2, nobs_per_subj*nsubj)) # `obstype=2` indicates panel data

model = multistatemodel(h12, h14, h15, h23, h25, h34, h35; data = dat)

simdat, paths = simulate(model; paths = true, data = true)

model = multistatemodel(h12, h14, h15, h23, h25, h34, h35; data = simdat[1])

set_crude_init!(model)


