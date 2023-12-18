# testing that a Markov likelihood with state censoring gives us the right answer
using DataFrames
using Distributions
using MultistateModels

# helpers
function observe_subjdat(path; censor = true)

    # censor?
    cens = (rand()[1] > 0.3 ? false : true) & censor
    
    # draw obstimes
    obstimes = [0.0; [1.0, 2.0] .+ round.(0.9 .* rand(2), sigdigits = 1)]

    if all(path.states .== 1)
        sdat = DataFrame(id = path.subj,
                         tstart = 0.0,
                         tstop = maximum(obstimes),
                         statefrom = 1,
                         stateto = 1,
                         obstype = 1)
    else
        # census path
        times_obs = sort(unique([obstimes; path.times[findall((path.states .== 4) .& (path.times .<= maximum(obstimes)))]]))
        obsinds  = searchsortedlast.(Ref(path.times), times_obs)
        states_obs = path.states[obsinds]

        if any(states_obs .== 4)
            inds = collect(1:findfirst(states_obs .== 4))
            times_obs = times_obs[inds]
            states_obs = states_obs[inds] 
        end
        
        if cens
            states_obs[findall(states_obs .∈ Ref([2,3]))] .= 0
        end

        sdat = DataFrame(id = path.subj,
                         tstart = times_obs[Not(end)],
                         tstop = times_obs[Not(begin)],
                         statefrom = states_obs[Not(end)],
                         stateto = states_obs[Not(begin)],
                         obstype = 2)

        # censor
        if any(sdat.stateto .== 4)
            insert!(sdat, nrow(sdat), sdat[nrow(sdat),:])
            sdat.tstop[nrow(sdat)-1] = sdat.tstop[nrow(sdat)-1] - sqrt(eps())
            sdat.tstart[nrow(sdat)] = sdat.tstop[nrow(sdat)-1]
            sdat.stateto[nrow(sdat)-1] = cens ? 0 : path.states[2]
            sdat.statefrom[nrow(sdat)] = cens ? 0 : path.states[2]
            sdat.obstype[nrow(sdat) - 1] = cens ? 3 : 2
            sdat.obstype[nrow(sdat)] = 1
        end

        sdat.obstype[findall(sdat.stateto .== 0)] .= 3
        sdat.obstype[findall(sdat.stateto .== 4)] .= 1
    end

    return sdat
end

function observe_subjdat2(path; censor = true)

    # censor?
    cens = (rand()[1] > 0.3 ? false : true) & censor
    
    # draw obstimes
    obstimes = [0.0; [1.0, 2.0] .+ round.(0.9 .* rand(2), sigdigits = 1)]

    if all(path.states .== 1)
        sdat = DataFrame(id = path.subj,
                         tstart = 0.0,
                         tstop = maximum(obstimes),
                         statefrom = 1,
                         stateto = 1,
                         obstype = 1)
    else
        # census path
        times_obs = sort(unique([obstimes; path.times[findall((path.states .== 3) .& (path.times .<= maximum(obstimes)))]]))
        obsinds  = searchsortedlast.(Ref(path.times), times_obs)
        states_obs = path.states[obsinds]

        if any(states_obs .== 3)
            inds = collect(1:findfirst(states_obs .== 3))
            times_obs = times_obs[inds]
            states_obs = states_obs[inds] 
        end

        sdat = DataFrame(id = path.subj,
                         tstart = times_obs[Not(end)],
                         tstop = times_obs[Not(begin)],
                         statefrom = states_obs[Not(end)],
                         stateto = states_obs[Not(begin)],
                         obstype = 2)

        # censor
        # if any(sdat.stateto .== 3)
        #     insert!(sdat, nrow(sdat), sdat[nrow(sdat),:])
        #     sdat.tstop[nrow(sdat)-1] = sdat.tstop[nrow(sdat)-1] - sqrt(eps())
        #     sdat.tstart[nrow(sdat)] = sdat.tstop[nrow(sdat)-1]
        #     sdat.stateto[nrow(sdat)-1] = cens ? 0 : path.states[2]
        #     sdat.statefrom[nrow(sdat)] = cens ? 0 : path.states[2]
        #     sdat.obstype[nrow(sdat) - 1] = cens ? 3 : 2
        #     sdat.obstype[nrow(sdat)] = 1
        # end

        sdat.obstype[findall(sdat.stateto .== 3)] .= 1
    end

    return sdat
end

# set up model
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
# h24 = Hazard(@formula(0 ~ 1), "exp", 2, 4)
# h34 = Hazard(@formula(0 ~ 1), "exp", 3, 4)

# set up dataset
nsubj = 1000

dat_sim = DataFrame(id = collect(1:nsubj),
                    tstart = 0.0,
                    tstop = 3.0,
                    statefrom = 1,
                    stateto = 2,
                    obstype = 1)

# make model and set parameters
# model_sim = multistatemodel(h12, h13, h24, h34; data = dat_sim)
model_sim = multistatemodel(h12, h23; data = dat_sim)

set_parameters!(model_sim,
               (h12 = [log(0.5),],
                h23 = [log(0.5),],
                h24 = [log(0.5),],
                h34 = [log(0.5),]))

# simulate
paths = simulate(model_sim; data = false, paths = true, nsim = 1)

# observe subject data
dat = reduce(vcat, [observe_subjdat2(p; censor = true) for p in paths])

# remake model object
censoring_patterns = [3 1 1 1 0;]
model_fit = multistatemodel(h12, h13, h24, h34; data = dat, CensoringPatterns = censoring_patterns)
# model_fit = multistatemodel(h12, h23; data = dat)

# fit model
set_crude_init!(model_fit)
fitted = fit(model_fit)

# simulate from fitted model
set_parameters!(model_sim, fitted.parameters)
paths_sim = simulate(model_sim; data = false, paths = true, nsim = 20)

mean(map(x -> any(x.states .== 4), paths_sim))
mean(map(x -> any(x.states .== 4), paths))


# input to the function `likelihood``
model=model_fit
books = MultistateModels.build_tpm_mapping(model.data)
parameters = MultistateModels.flatview(model.parameters)
data = MultistateModels.MPanelData(model, books)

# the function `likelihood` works
MultistateModels.loglik(parameters, data)


# but the optimizer does not work
optf = OptimizationFunction(MultistateModels.loglik, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, parameters, MultistateModels.MPanelData(model, books))
sol  = solve(prob, Newton())

