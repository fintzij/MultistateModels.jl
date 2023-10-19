using CSV
using DataFrames
using Distributions
using MultistateModels

# First fit a Weibull model to the ACTT-1 data to get reasonable parameters to simulate from
actt = CSV.read("/home/liangcj/scratch/actt1_mm.csv", DataFrame)

# rename states 1 and 2 as state 3 then subtract 2 from all states so states start at 1
# so 1: recovered, 2: hospitalized, 3: suppl O2, 4: NIPPV/high flow, 5: vent, 6: dead

# 1: recovered, 2: hospitalized no ICU, 3: ICU, 4: hospitalized with ICU, 5: dead

# in ICU 6/7, hosp w/ icu 4/5 following 6/7, hosp w/o icu 4/5, recover, dead
# constrain all trans to recover to same trt effect, all trans to death same trt effect

for i in unique(actt.id)
    icu = 0
    for j in findall(actt.id .== i)
        # patient has visited ICU
        if ((actt.statefrom[j] ∈ [6,7]) | (actt.stateto[j] ∈ [6,7]))
            icu = 1
        end
        newfrom = 0
        newto = 0

        # hospitalized with ICU history
        if ((actt.statefrom[j] ∈ [4,5]) & (icu == 1))
            newfrom = 4
        end
        if ((actt.stateto[j] ∈ [4,5]) & (icu == 1))
            newto = 4
        end

        # hospitalized with no ICU history
        if ((actt.statefrom[j] ∈ [4,5]) & (icu == 0))
            newfrom = 2
        end
        if ((actt.stateto[j] ∈ [4,5]) & (icu == 0))
            newto = 2
        end

        # ICU
        if (actt.statefrom[j] ∈ [6,7])
            newfrom = 3
        end
        if (actt.stateto[j] ∈ [6,7])
            newto = 3
        end

        # recovered
        if (actt.statefrom[j] ∈ [1,2])
            newfrom = 1
        end
        if (actt.stateto[j] ∈ [1,2])
            newto = 1
        end

        # dead
        if (actt.statefrom[j] == 8)
            newfrom = 5
        end
        if (actt.stateto[j] == 8)
            newto = 5
        end

        actt.statefrom[j] = newfrom
        actt.stateto[j] = newto
    end
end


#= jon's version of preprocessing
for s in unique(actt.id)
    subj_dat = @view actt[findall(actt.id .== s), :]

    # replace post-ICU hospitalized states
    if any(subj_dat.stateto .∈ Ref([6,7])) 
        # find last index of 6 or 7
        lasticu = findlast(subj_dat.stateto .∈ Ref([6,7]))
        subj_dat.stateto[findall(subj_dat.stateto[lasticu:nrow(subj_dat)] .∈ Ref([4,5]))] .= -4
    end

    # replace other non-ICU hospitalized states
    subj_dat.stateto[findall(subj_dat.stateto .∈ Ref([4,5]))] .= -2

    # replace ICU states
    subj_dat.stateto[findall(subj_dat.stateto .∈ Ref([6,7]))] .= -3

    # replace recovered states
    subj_dat.stateto[findall(subj_dat.stateto .∈ Ref([1,2,3]))] .= -1

    # recode death
    subj_dat.stateto[findall(subj_dat.stateto .== 8)] .= -5

    # make positive
    subj_dat.stateto .= -subj_dat.stateto

    # fill in statefrom
    subj_dat.statefrom[Not(1)] = subj_dat.stateto[Not(last)]
end
 =#
 
# fit exponential model with no covariates
h21 = Hazard(@formula(0~1), "exp", 2, 1) # hosp to recovered
h23 = Hazard(@formula(0~1), "exp", 2, 3) # hosp to suppl O2
h31 = Hazard(@formula(0~1), "exp", 3, 1) # suppl O2 to recovered
h32 = Hazard(@formula(0~1), "exp", 3, 2) # suppl O2 to hosp
h34 = Hazard(@formula(0~1), "exp", 3, 4) # suppl O2 to NIPPV/high flow
h35 = Hazard(@formula(0~1), "exp", 3, 5) # suppl O2 to vent
h36 = Hazard(@formula(0~1), "exp", 3, 6) # suppl O2 to death
h43 = Hazard(@formula(0~1), "exp", 4, 3) # NIPPV/high flow to suppl O2
h45 = Hazard(@formula(0~1), "exp", 4, 5) # NIPPV/high flow to vent
h46 = Hazard(@formula(0~1), "exp", 4, 6) # NIPPV/high flow to death
h54 = Hazard(@formula(0~1), "exp", 5, 4) # vent to NIPPV/high flow
h56 = Hazard(@formula(0~1), "exp", 5, 6) # vent to death

model0 = multistatemodel(h21, h23, # no covariates
                         h31, h32, h34, h35, h36, 
                         h43, h45, h46, 
                         h54, h56; 
                         data = actt)
fit_model0 = fit(model0)

#= julia> fit_model0.parameters
12-element ArraysOfArrays.VectorOfVectors{Float64, Vector{Float64}, Vector{Int64}, Vector{Tuple{}}}:
 [-1.6450680028426914]
 [-2.3009179711198473]
 [-2.292491138479795]
 [-2.323885063284237]
 [-2.75893996631064]
 [-4.250950463115472]
 [-67.04089155587138]
 [-1.2913707652426596]
 [-2.509691005889884]
 [-3.812638893004995]
 [-2.872776724451048]
 [-4.310978344694081] =#

# fit exponential model with trt covariates
h21 = Hazard(@formula(0~1+trt), "exp", 2, 1) # hosp to recovered
h23 = Hazard(@formula(0~1+trt), "exp", 2, 3) # hosp to suppl O2
h31 = Hazard(@formula(0~1+trt), "exp", 3, 1) # suppl O2 to recovered
h32 = Hazard(@formula(0~1+trt), "exp", 3, 2) # suppl O2 to hosp
h34 = Hazard(@formula(0~1+trt), "exp", 3, 4) # suppl O2 to NIPPV/high flow
h35 = Hazard(@formula(0~1+trt), "exp", 3, 5) # suppl O2 to vent
h36 = Hazard(@formula(0~1+trt), "exp", 3, 6) # suppl O2 to death
h43 = Hazard(@formula(0~1+trt), "exp", 4, 3) # NIPPV/high flow to suppl O2
h45 = Hazard(@formula(0~1+trt), "exp", 4, 5) # NIPPV/high flow to vent
h46 = Hazard(@formula(0~1+trt), "exp", 4, 6) # NIPPV/high flow to death
h54 = Hazard(@formula(0~1+trt), "exp", 5, 4) # vent to NIPPV/high flow
h56 = Hazard(@formula(0~1+trt), "exp", 5, 6) # vent to death

model_trt = multistatemodel(h21, h23, # no covariates
                         h31, h32, h34, h35, h36, 
                         h43, h45, h46, 
                         h54, h56; 
                         data = actt)
fit_model_trt = fit(model_trt)

#= julia> fit_model_trt.parameters
12-element ArraysOfArrays.VectorOfVectors{Float64, Vector{Float64}, Vector{Int64}, Vector{Tuple{}}}:
 [-1.614154125644098, -0.05749997599121682]
 [-2.1516510696161824, -0.2975281888308755]
 [-2.363020023801264, 0.1320331790262221]
 [-2.2725189591085995, -0.10324775782954772]
 [-2.7069659224954683, -0.3149234360887575]
 [-4.255732703007216, -0.08693543266987477]
 [-4.878649286345931, -0.7219442409680682]
 [-1.2800392122805277, -0.027919027659653116]
 [-2.350165523221071, -0.36240196415128906]
 [-4.713461286304682, 0.4700825100465047]
 [-2.896648964642095, 0.04576695816643554]
 [-4.31688030902105, -0.01566918466640339] =#


# fit Weibull model with no covariates
h21 = Hazard(@formula(0~1), "wei", 2, 1) # hosp to recovered
h23 = Hazard(@formula(0~1), "wei", 2, 3) # hosp to suppl O2
h31 = Hazard(@formula(0~1), "wei", 3, 1) # suppl O2 to recovered
h32 = Hazard(@formula(0~1), "wei", 3, 2) # suppl O2 to hosp
h34 = Hazard(@formula(0~1), "wei", 3, 4) # suppl O2 to NIPPV/high flow
h35 = Hazard(@formula(0~1), "wei", 3, 5) # suppl O2 to vent
h36 = Hazard(@formula(0~1), "wei", 3, 6) # suppl O2 to death
h43 = Hazard(@formula(0~1), "wei", 4, 3) # NIPPV/high flow to suppl O2
h45 = Hazard(@formula(0~1), "wei", 4, 5) # NIPPV/high flow to vent
h46 = Hazard(@formula(0~1), "wei", 4, 6) # NIPPV/high flow to death
h54 = Hazard(@formula(0~1), "wei", 5, 4) # vent to NIPPV/high flow
h56 = Hazard(@formula(0~1), "wei", 5, 6) # vent to death

model_wei = multistatemodel(h21, h23, # trt covariate
                         h31, h32, h34, h35, h36, 
                         h43, h45, h46, 
                         h54, h56; 
                         data = actt)
fit_model_wei = fit(model_wei)

#= julia> fit_model_wei.parameters
12-element ArraysOfArrays.VectorOfVectors{Float64, Vector{Float64}, Vector{Int64}, Vector{Tuple{}}}:
 [0.055010057514710255, -1.8200157697630284]
 [-0.3785429034666941, 0.03206872371197693]
 [0.036952653449049036, -2.1281497812352654]
 [0.029959269002226986, -0.3669839604271485]
 [-0.14494363311488867, -0.6848900338004505]
 [-0.36645981737973793, -1.8371446410696595]
 [0.8361272628015204, -5.882505083337275]
 [-0.25522462405341517, -0.02812042063957907]
 [-0.3148567609006092, -0.2606481415794421]
 [-0.09355292608079392, -4.350586090671693]
 [-0.4268046705343594, -0.4101759827554782]
 [0.15649504059554772, -4.591919762963543]
 =#

# fit Weibull model with treatment covariate
h21 = Hazard(@formula(0~1+trt), "wei", 2, 1) # hosp to recovered
h23 = Hazard(@formula(0~1+trt), "wei", 2, 3) # hosp to suppl O2
h31 = Hazard(@formula(0~1+trt), "wei", 3, 1) # suppl O2 to recovered
h32 = Hazard(@formula(0~1+trt), "wei", 3, 2) # suppl O2 to hosp
h34 = Hazard(@formula(0~1+trt), "wei", 3, 4) # suppl O2 to NIPPV/high flow
h35 = Hazard(@formula(0~1+trt), "wei", 3, 5) # suppl O2 to vent
h36 = Hazard(@formula(0~1+trt), "wei", 3, 6) # suppl O2 to death
h43 = Hazard(@formula(0~1+trt), "wei", 4, 3) # NIPPV/high flow to suppl O2
h45 = Hazard(@formula(0~1+trt), "wei", 4, 5) # NIPPV/high flow to vent
h46 = Hazard(@formula(0~1+trt), "wei", 4, 6) # NIPPV/high flow to death
h54 = Hazard(@formula(0~1+trt), "wei", 5, 4) # vent to NIPPV/high flow
h56 = Hazard(@formula(0~1+trt), "wei", 5, 6) # vent to death

model_wei_trt = multistatemodel(h21, h23, # trt covariate
                         h31, h32, h34, h35, h36, 
                         h43, h45, h46, 
                         h54, h56; 
                         data = actt)
fit_model_wei_trt = fit(model_wei_trt)

#= julia> fit_model_wei_trt.parameters
12-element ArraysOfArrays.VectorOfVectors{Float64, Vector{Float64}, Vector{Int64}, Vector{Tuple{}}}:
 [0.06645255524659988, -1.8095563893888695, 0.054707996053257865]
 [-0.3616378144132654, 0.038582563600487736, -0.0442020748318712]
 [0.06058876934488346, -2.185719702312976, 0.05228906482000337]
 [-0.016451690991205855, -0.3059523776593508, 0.032190528958703536]
 [-0.08567478852035226, -0.5734926849535077, -0.02539597983770484]
 [-0.26033250916177086, -1.5953406111880986, -0.21866475442308855]
 [-0.21338806919725076, -4.493748487428925, -0.388682699061942]
 [-0.20752263615749317, -0.030502529002255474, 0.035599644294233214]
 [-0.29149904116892844, -0.15967503069858202, -0.10457529090874552]
 [0.4208078415737228, -5.336023323433253, 0.46057715736156846]
 [-0.390024361873267, -0.4050468141300625, 0.07767639734926303]
 [0.14155130499042634, -4.529679261878313, -0.05654464793773927] =#




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


