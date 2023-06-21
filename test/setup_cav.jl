using MultistateModels
using CSV
using DataFrames

cav = CSV.read("scratch/cav.csv", DataFrame)

subjids = unique(cav.PTNUM)

cav_startstop = DataFrame(
    id=zeros(0),
    tstart=zeros(0),
    tstop=zeros(0),
    statefrom=zeros(0),
    stateto=zeros(0),
    obstype=zeros(0),
    sex=zeros(0)
)

for subjid in subjids

    # subject data
    cav_subj = filter(row -> row.PTNUM == subjid, cav)
    n_obs    = nrow(cav_subj) # number of observations

    tstart    = cav_subj.years[1:n_obs-1] # remove last time
    tstop     = cav_subj.years[2:n_obs  ] # remove first time
    statefrom = cav_subj.state[1:n_obs-1]
    stateto   = cav_subj.state[2:n_obs  ]
    obstype   = cav_subj.obstype[2:n_obs  ] # obstype value for the first observation from each subject is not used

    # start-stop format
    n_trans = n_obs - 1 # number of transitions
    append!(
        cav_startstop,
        DataFrame(
            id = fill(subjid, n_trans),
            tstart=tstart,
            tstop=tstop,
            statefrom=statefrom,
            stateto=stateto,
            obstype=obstype,
            sex=fill(cav_subj.sex[1], n_trans)
            )
        )
end

h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
h31 = Hazard(@formula(0 ~ 1), "exp", 3, 1)
h14 = Hazard(@formula(0 ~ 1), "exp", 1, 4)
#h41 = Hazard(@formula(0 ~ 1), "exp", 4, 1) # state 4 is absorbing
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
h32 = Hazard(@formula(0 ~ 1), "exp", 3, 2)
h24 = Hazard(@formula(0 ~ 1), "exp", 2, 4)
#h42 = Hazard(@formula(0 ~ 1), "exp", 4, 2)
h34 = Hazard(@formula(0 ~ 1), "exp", 3, 4)
#h43 = Hazard(@formula(0 ~ 1), "exp", 4, 3)

m = multistatemodel(h12, h21, h13, h31, h14, h23, h32, h24, h34; data = cav_startstop)

set_parameters!(m, 
    (h12 = [log(0.068),], # values found by msm::crudeinits()
     h13 = [log(0.015),],
     h14 = [log(0.049),],
     h21 = [log(0.117),],
     h23 = [log(0.137),],
     h24 = [log(0.122),],
     h31 = [log(0.015),],
     h32 = [log(0.049),],
     h34 = [log(0.208),]))


m_fit = fit(m)

MultistateModels.getloglik(m_fit)
MultistateModels.getestimates(m_fit)
MultistateModels.getestimates(m_fit; transformed = true)

# todo

## profile code, identify bottlnecks

## compare runtime with R package msm