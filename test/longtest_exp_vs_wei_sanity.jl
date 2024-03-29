using DataFrames
using Distributions
using MultistateModels
using MultistateModels: loglik
using Plots

# helpers
function make_obstimes()    
    times = collect(0:0.25:1) .+ vcat([0.0; (rand(Beta(5,5), 4) * 0.2 .- 0.1)])    
    return times
end

function makepars()
    parameters = (h12 = [log(1.5), log(1.5)],
                  h13 = [log(2/3), log(2/3)],
                  h23 = [log(2), log(3)])
    return parameters
end

# observe data
function observe_dat(dat_raw, paths)
    
    # initialize
    dat = similar(dat_raw, 0)

    # group dat_raw
    gdat = groupby(dat_raw, :id)
    
    # observe dat
    for p in eachindex(paths)
        if !(3 ∈ paths[p].states)
            append!(dat, gdat[p])

        else
            # insert row and change the time of death to exact
            subjdat = copy(gdat[p])
            insert!(subjdat, nrow(subjdat), subjdat[nrow(subjdat),:])
            subjdat.tstop[end] = last(paths[p].times)
            subjdat.obstype[end] = 1
            subjdat.tstart[end] = subjdat.tstop[end] - sqrt(eps())
            subjdat.tstop[nrow(subjdat)-1] = subjdat.tstart[end]
            subjdat.stateto[nrow(subjdat)-1] = subjdat.statefrom[end]
            
            # append
            append!(dat, subjdat)
        end
    end

    return dat
end

# set up dataset
nsubj = 1000
ntimes = 4
visitdays = [make_obstimes() for i in 1:nsubj]
data = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
            tstart = reduce(vcat, map(x -> x[Not(end)], visitdays)),
            tstop = reduce(vcat, map(x -> x[Not(1)], visitdays)),
            statefrom = fill(1, nsubj * ntimes),
            stateto = fill(1, nsubj * ntimes),
            obstype = fill(2, nsubj * ntimes))

# transition intensities
h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h13_exp = Hazard(@formula(0 ~ 1), "exp", 1, 3)
h23_exp = Hazard(@formula(0 ~ 1), "exp", 2, 3)

h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h13_wei = Hazard(@formula(0 ~ 1), "wei", 1, 3)
h23_wei = Hazard(@formula(0 ~ 1), "wei", 2, 3)

# models
model_sim = multistatemodel(h12_exp, h13_exp, h23_exp; data = data)
set_parameters!(model_sim,
                (h12 = [log(1.5),],
                 h13 = [log(2/3),],
                 h23 = [log(3),]))

# simulate data
dat_raw, paths = simulate(model_sim; paths = true, data = true)
# dat = observe_dat(dat_raw[1], paths)

# remake models
# model_exp = multistatemodel(h12_exp, h13_exp, h23_exp; data = dat)

# constraints = make_constraints(
#     cons = [:(h12_shape;), :(h13_shape;), :(h23_shape;)],
#     lcons = [0.0, 0.0, 0.0],
#     ucons =[0.0, 0.0, 0.0])

# model_wei = multistatemodel(h12_wei, h13_wei, h23_wei; data = dat)

# # fit - CHECKS OUT
# fit_exp = fit(model_exp)
# fit_wei = fit(model_wei; constraints = constraints, ess_target_initial = 100)

# remake data to use full panel data
model_exp = multistatemodel(h12_exp, h13_exp, h23_exp; data = dat_raw[1])

constraints = make_constraints(
    cons = [:(h12_shape;), :(h13_shape;), :(h23_shape;)],
    lcons = [0.0, 0.0, 0.0],
    ucons =[0.0, 0.0, 0.0])

model_wei = multistatemodel(h12_wei, h13_wei, h23_wei; data = dat_raw[1])

# fit - CHECKS OUT
fit_exp = fit(model_exp)
fit_wei = fit(model_wei; constraints = constraints, ess_target_initial = 100)

paths_exp = simulate(fit_exp; paths = true, data = false, nsim = 500)
paths_wei = simulate(fit_wei; paths = true, data = false, nsim = 500)

# compute log like
subj_ll_exp = zeros(size(paths_exp, 2))
subj_ll_wei = zeros(size(paths_wei, 2))

for k in 1:size(paths_exp, 2)
    for j in 1:size(paths_exp, 1)
        subj_ll_exp[k] += loglik(fit_exp.parameters, paths_exp[j,k], fit_exp.hazards, fit_exp)
        subj_ll_wei[k] += loglik(fit_wei.parameters, paths_exp[j,k], fit_wei.hazards, fit_wei)
    end
end

# pretty close
mean(subj_ll_exp)
mean(subj_ll_wei)


# compare subj_ll
# sort(abs.(fit_exp.subj_ll .- fit_wei.subj_ll))
# mean(abs.(fit_exp.subj_ll .- fit_wei.subj_ll))