### Functions
function make_censorings()
    censoring_patterns = zeros(Int64, 5,10)
    censoring_patterns[:,1]        .= 3:7 
    censoring_patterns[1,[2,5,6]] .= 1
    censoring_patterns[2,[3,4]]   .= 1
    censoring_patterns[3,[5,6]]   .= 1
    censoring_patterns[4,[7,8]]   .= 1
    censoring_patterns[5,[9,10]]  .= 1
    return censoring_patterns
end

# function to make the parameters
function makepars(; nulleff = 1)
    te = 2 - nulleff == 1
    parameters = (h12 = [log(0.4), log(0.2), log(0.65) * te],
                 h23 = [log(2), log(0.2), log(0.5) * te],
                 h24 = [log(2.5), log(0.3), log(1) * te],
                 h35 = [log(2.5), log(0.3), log(1) * te],
                 h45 = [log(2), log(0.25), log(0.5) * te],
                 h26 = [log(0.4), log(1), log(0.66) * te],
                 h67 = [log(2), log(0.2), log(0.35) * te],
                 h68 = [log(2), log(0.3), log(1) * te],
                 h79 = [log(2), log(0.3), log(1) * te],
                 h89 = [log(2), log(0.55), log(0.35) * te])
    return parameters
end

# function to make the PCR assessment times
function makepcrs(;ntimes=1)
    if ntimes == 4
        times = [0.0, 1.0, 2.0, 3.0, 4.0] .+ [0.0; sample(collect(-2.0:2.0)/7, Weights([0.075, 0.2, 0.55, 0.1, 0.075]), 4)]
    elseif ntimes == 1
        times = [0.0, 4.0] .+ [0.0; sample(collect(-2.0:2.0)/7, Weights([0.075, 0.2, 0.55, 0.1, 0.075]), 1)]
    end
    return times
end

# function to set up the model
function setup_model(; make_pars, data = nothing, SamplingWeights = nothing, n_per_arm = 750, ntimes = 4, family = "wei", nulleff = 1)
    
    # create hazards
    h12 = Hazard(@formula(0 ~ 1 + mab), family, 1, 2)
    h23 = Hazard(@formula(0 ~ 1 + mab), family, 2, 3)
    h24 = Hazard(@formula(0 ~ 1 + mab), family, 2, 4)
    h35 = Hazard(@formula(0 ~ 1 + mab), family, 3, 5)
    h45 = Hazard(@formula(0 ~ 1 + mab), family, 4, 5)
    h26 = Hazard(@formula(0 ~ 1 + mab), family, 2, 6)
    h67 = Hazard(@formula(0 ~ 1 + mab), family, 6, 7)
    h68 = Hazard(@formula(0 ~ 1 + mab), family, 6, 8)
    h79 = Hazard(@formula(0 ~ 1 + mab), family, 7, 9)
    h89 = Hazard(@formula(0 ~ 1 + mab), family, 8, 9)

    # data for simulation parameters
    if isnothing(data)
        visitdays = [makepcrs(;ntimes = ntimes) for i in 1:(2*n_per_arm)]
        data = DataFrame(id = repeat(collect(1:(2 * n_per_arm )), inner = ntimes),
                    tstart = reduce(vcat, map(x -> x[Not(end)], visitdays)),
                    tstop = reduce(vcat, map(x -> x[Not(1)], visitdays)),
                    statefrom = fill(1, 2 * n_per_arm * ntimes),
                    stateto = fill(1, 2 * n_per_arm * ntimes),
                    obstype = fill(1, 2 * n_per_arm * ntimes),
                    mab = repeat([0.0, 1.0], inner = n_per_arm * ntimes))
    end

    if isnothing(SamplingWeights)
        SamplingWeights = ones(Float64, n_per_arm * 2)
    end
    
    # create model
    model = multistatemodel(h12, h23, h24, h35, h45, h26, h67, h68, h79, h89; data = data, CensoringPatterns = make_censorings(), SamplingWeights = SamplingWeights)

    # set parameters
    if make_pars
        parameters = makepars(; nulleff = nulleff)
        set_parameters!(model, parameters) 
    else
        set_crude_init!(model)
    end

    # return model
    return model
end

# function to censor subject data
function censor_subjdat!(subjdat; censmat = make_censorings())
    
    # summarize history
    seroconv      = any(subjdat.stateto .∈ Ref([3,5,7,9]))
    eversymp      = any(subjdat.stateto .> 5)
    during_pcrpos = findall(subjdat.pcr .== 1)
    
    if length(during_pcrpos) > 0
        before_pcrpos = collect(range(1, length = first(during_pcrpos) - 1))
        after_pcrpos = collect(range(last(during_pcrpos) + 1, length = nrow(subjdat) - last(during_pcrpos)))
    else
        before_pcrpos = collect(1:nrow(subjdat))
        after_pcrpos = Int64[]
    end

    # branch on serostatus and ever symtomatic
    if seroconv & !eversymp
        # censoring codes into stateto
        if length(during_pcrpos) == 0
            subjdat.stateto .= -3
        else
            subjdat.stateto[before_pcrpos] .= 1
            subjdat.stateto[during_pcrpos] .= -4
            subjdat.stateto[after_pcrpos]  .= -5
        end

        # state at serology assessment is known
        subjdat.stateto[findall((subjdat.sero .== 1) .& (subjdat.pcr .== 1))] .= 3
        subjdat.stateto[findall((subjdat.sero .== 1) .& (subjdat.pcr .== 0))] .= 5

    elseif seroconv & eversymp
        # find symptom onset
        onset    = findfirst(subjdat.covid .== 1)
        presymp  = collect(range(1, length = onset - 1))
        postsymp = collect(range(onset + 1, length = nrow(subjdat) - onset))

        # assign states and censoring codes
        subjdat.stateto[before_pcrpos] .= 1
        subjdat.stateto[setdiff(presymp, before_pcrpos)] .= 2
        subjdat.stateto[onset] = 6
        subjdat.stateto[intersect(postsymp, during_pcrpos)] .= -6
        subjdat.stateto[intersect(postsymp, after_pcrpos)] .= -7

        # state at serology assessment is known
        subjdat.stateto[end] = last(subjdat.pcr) == 1 ? 7 : 9
    end

    # assign states from
    subjdat.statefrom[Not(1)] .= subjdat.stateto[Not(end)]

    # censoring patterns
    cens = findall(subjdat.stateto .< 0)
    subjdat.obstype[cens]  = -subjdat.stateto[cens]
    subjdat.stateto[cens] .= 0

    # double up on assigning states from to states to
    subjdat.statefrom[Not(1)] .= subjdat.stateto[Not(end)]

    # return the subject dataframe
    return subjdat
end

# function to observe a sample paths at a vector of observation times
function observe_path(samplepath, times)
    samplepath.states[searchsortedlast.(Ref(samplepath.times), times)]
end

# function to continuously observe a sample path
function observe_subjpath(path, model)
    ntimes  = length(path.times) - 1 
    subjdat = DataFrame(
                id = fill(path.subj, ntimes),
                tstart = path.times[Not(end)],
                tstop = path.times[Not(1)],
                statefrom = path.states[Not(end)],
                stateto = path.states[Not(1)],
                obstype = fill(1, ntimes),
                mab = fill(model.data.mab[findfirst(model.data.id .== path.subj)], ntimes))
    return subjdat
end

# function to wrangle a path to subject data
function observe_subjdat(path, model; censor = false)

    # grab subject's data
    i = path.subj
    subj_dat = model.data[model.subjectindices[i], :]

    # state sequence at observation times
    times  = [0.0; subj_dat.tstop]
    if any(path.states .== 6)
        infecind = findfirst(path.states .== 6)
        delta = (path.times[infecind] - path.times[infecind-1]) < sqrt(eps()) ? path.times[infecind] - path.times[infecind-1] : sqrt(eps())
        push!(times, path.times[infecind] - delta)
        push!(times, path.times[infecind])
        sort!(times)
    end
    states = observe_path(path, times)

    # data frame
    subjdat = DataFrame(id = fill(path.subj, length(times) - 1),
                        tstart = times[Not(end)],
                        tstop = times[Not(1)],
                        statefrom = states[Not(end)],
                        stateto = states[Not(1)],
                        obstype = fill(2, length(times) - 1),
                        mab = fill(subj_dat.mab[1], length(times) - 1),
                        pcr = ifelse.(states[Not(1)] .∈ Ref([2,3,6,7]), 1, 0),
                        covid = ifelse.(states[Not(1)] .> 5, 1, 0),
                        sero = ifelse.(states[Not(1)] .∈ Ref([3,5,7,9]), 1, 0))

    # correct obstype for symptomatic covid
    if any(subjdat.stateto .== 6)
        subjdat.obstype[findfirst(subjdat.stateto .== 6)] = 1
    end

    return censor ? censor_subjdat!(subjdat) : subjdat
end

# helper to get the duration of PCR+
function pcrpos(path, subjdat)
    tmax = last(subjdat.tstop)
    posinds = findall(path.states .∈ Ref([2,3,6,7]))
    length(posinds) > 0 ? min(tmax, path.times[last(posinds)]) - path.times[first(posinds)] : missing
end

# summarize paths
function summarize_paths(paths, dat)

    # get mab assignments
    mab = @chain dat begin
        groupby(:id)
        combine(:mab => unique)
        rename([2 => :mab])
    end

    # views
    mabinds    = findall(mab.mab .== 1)
    placinds   = findall(mab.mab .== 0)
    paths_mab  = view(paths, mabinds, :)
    paths_plac = view(paths, placinds, :)

    # find infections
    infected_mab  = map(x -> any(x.states .== 2), paths_mab)
    infected_plac = map(x -> any(x.states .== 2), paths_plac)

    # find symptomatic infections
    sympt_mab  = map(x -> any(x.states .== 6), paths_mab)
    sympt_plac = map(x -> any(x.states .== 6), paths_plac)

    # find sero+ 
    sero_symp_mab   = map(x -> any(x.states .∈ Ref([7,9])), paths_mab)
    sero_symp_plac  = map(x -> any(x.states .∈ Ref([7,9])), paths_plac)
    sero_asymp_mab  = map(x -> any(x.states .∈ Ref([3,5])), paths_mab)
    sero_asymp_plac = map(x -> any(x.states .∈ Ref([3,5])), paths_plac)

    # event times
    infectimes_mab  = map(x -> any(x.states .== 2) ? x.times[findfirst(x.states .== 2)] : missing, paths_mab)  
    infectimes_plac = map(x -> any(x.states .== 2) ? x.times[findfirst(x.states .== 2)] : missing, paths_plac)  

    symptimes_mab  = map(x -> any(x.states .== 6) ? x.times[findfirst(x.states .== 6)] : missing, paths_mab)  
    symptimes_plac = map(x -> any(x.states .== 6) ? x.times[findfirst(x.states .== 6)] : missing, paths_plac)

    pcrdurs = [pcrpos(paths[x,y], view(dat, findall(dat.id .== x), :)) for x in 1:size(paths, 1), y in 1:size(paths, 2)]

    # results
    # m: number of mabs
    # p: number of placebos
    # i_m: number infected mab
    # i_p: number infected placebos
    # s_i_m: number of symptomatic mabs
    # s_i_p: number of symptomatic placebo
    # n_s_m: number N+ among symptomatic mabs
    # n_s_p: number N+ among symptomatic placebo
    # n_as_m: number N+ among asymptomatic mabs
    # n_as_m: number N+ among asymptomatic placebo
    # ts_m: mean infection onset time, mab
    # ts_p: mean infection onset time, placebo
    # ts_m: mean symptom onset time, mab
    # ts_p: mean symptom onset time, placebo
    # pcrdur_m: mean duration of PCR+, mab
    # pcrdur_p: mean duration of PCR+, placebo
    m      = length(paths_mab)    
    p      = length(paths_plac)   
    i_m    = sum(infected_mab)    
    i_p    = sum(infected_plac)   
    s_i_m  = sum(sympt_mab)       
    s_i_p  = sum(sympt_plac)      
    n_s_m  = sum(sero_symp_mab)   
    n_s_p  = sum(sero_symp_plac)
    n_as_m = sum(sero_asymp_mab) 
    n_as_p = sum(sero_asymp_plac) 

    ests = (i_m      = i_m / m, 
            i_p      = i_p / p,
            s_i_m    = s_i_m / i_m,
            s_i_p    = s_i_p / i_p,
            n_s_m    = n_s_m / i_m,
            n_as_m   = n_as_m / i_m,
            n_s_p    = n_s_p / i_p,
            n_as_p   = n_as_p / i_p,
            ti_m     = !all(ismissing.(infectimes_mab)) ? mean(filter(x -> !ismissing(x), infectimes_mab)) : -1.0, 
            ti_p     = !all(ismissing.(infectimes_plac)) ? mean(filter(x -> !ismissing(x), infectimes_plac)) : -1.0,
            ts_m     = !all(ismissing.(symptimes_mab)) ? mean(filter(x -> !ismissing(x), symptimes_mab)) : -1.0,
            ts_p     = !all(ismissing.(symptimes_plac)) ? mean(filter(x -> !ismissing(x), symptimes_plac)) : -1.0,
            pcrdur_m = !all(ismissing.(view(pcrdurs, mabinds, :))) ? (mean(filter(x -> !ismissing(x), view(pcrdurs, mabinds, :)))) : -1.0,
            pcrdur_p = !all(ismissing.(view(pcrdurs, placinds, :))) ? (mean(filter(x -> !ismissing(x), view(pcrdurs, placinds, :)))) : -1.0)

    return ests
end

# asymptotic bootstrap 
function asymptotic_bootstrap(model, pars, vcov, sims_per_subj, nboot)

    # draw parameters
    pars = flatview(pars)
    npars = length(pars)
    pardraws = zeros(Float64, npars)

    # SVD
    U = zeros(npars, npars) 
    D = zeros(npars)
    U,D = svd(vcov)

    # replace small negative singular values with zeros
    D[findall(D .< 0)] .= 0.0

    # matrix square root
    S = U * diagm(sqrt.(D))
    
    # initialize matrix of estimates
    ests = zeros(Float64, 14, nboot)

    # simulate paths under each set of parameters
    for k in 1:nboot
        # draw parameters
        pardraws[1:npars] = flatview(pars) .+ S * randn(npars)

        # set the parameters
        set_parameters!(model, VectorOfVectors(pardraws, model.parameters.elem_ptr))

        # simulate paths
        paths_sim = simulate(model; nsim = sims_per_subj, paths = true, data = false)

        # summarize paths
        ests[:,k] = collect(summarize_paths(paths_sim, model.data))
    end

    return mapslices(x -> quantile(x[findall(map(y -> (!ismissing(y) && (y != -1.0)), x))], [0.025, 0.975]), ests, dims = [2,])
end

# Bayesian bootstrap
function bayesian_bootstrap(dat, pars, sims_per_subj, nboot)

    # get mab assignments
    mab = @chain dat begin
        groupby(:id)
        combine(:mab => unique)
        rename([2 => :mab])
    end

    # prepare to sample weights
    nsubj = nrow(mab)
    nmab  = sum(mab.mab)
    nplac = nsubj - nmab

    mabinds  = findall(mab.mab .== 1)
    placinds = findall(mab.mab .== 0)

    # initialize matrix for results
    boots = Array{Union{Missing, Float64}}(missing, 14, nboot)

    # fit lots and lots of models
    Threads.@threads for k in 1:nboot

        @info "Bootstrap iteration $k of $nboot"

        # reweight
        weights = ones(Float64, nsubj)
        weights[mabinds]  .= nmab .* rand(Dirichlet(ones(Int64(nmab))), 1)[:,1]
        weights[placinds] .= nplac .* rand(Dirichlet(ones(Int64(nplac))), 1)[:,1]

        dat_boot, weights_boot = collapse_data(dat; SamplingWeights = weights)

        # remake the model
        model_fit_boot = setup_model(; make_pars = false, data = dat_boot, SamplingWeights = weights_boot)
        model_sim_boot = setup_model(; make_pars = false, data = dat)

        # set parameters
        set_parameters!(model_fit_boot, pars)

        # fit the model and get estimates
        ests_boot = try
            # fit model
            fitted_boot = fit(model_fit_boot; verbose = false, compute_vcov = false)

            # set the parameters for simulation
            set_parameters!(model_sim_boot, fitted_boot.parameters)

            # simulate paths
            paths_boot = simulate(model_sim_boot; nsim = sims_per_subj, paths = true, data = false)

            # summarize paths
            summarize_paths(paths_boot, model_sim_boot.data)
        catch err
            fill(missing, 14)
        end

        # assign to boots
        boots[:,k] = collect(ests_boot)
    end

    # return CIs
    return permutedims(mapslices(x -> quantile(x[Not(findall(ismissing.(x)))], [0.025, 0.975]), Matrix(boots), dims = [1,]))
end

# wrapper for one simulation
# nsim is the number of simulated paths per subject
# ndraws is the number of draws from the asymptotic normal distribution of the MLEs
function work_function(;simnum, seed, cens, nulleff, sims_per_subj, nboot)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", n_per_arm = 750, nulleff = nulleff)
        
    # simulate paths
    paths = simulate(model_sim; nsim = 1, paths = true, data = false)[:,1]

    # make dataset
    dat = cens == 1 ? reduce(vcat, map(x -> observe_subjpath(x, model_sim), paths)) : cens == 2 ? reduce(vcat, map(x -> observe_subjdat(x, model_sim; censor = false), paths)) : reduce(vcat, map(x -> observe_subjdat(x, model_sim; censor = true), paths))

    ### set up model for fitting
    dat_collapsed, weights = collapse_data(dat)
    model_fit = setup_model(; make_pars = false, data = dat_collapsed, SamplingWeights = weights, family = "wei")

    # fit model
    model_fitted = cens == 1 ? fit(model_fit; verbose = true, compute_vcov = true) : fit(model_fit; verbose = true, compute_vcov = true, maxiter = 250)

    # move the parameters over to the model for simulation
    set_parameters!(model_sim, model_fitted.parameters)

    ### simulate from the fitted model
    paths_sim = simulate(model_sim; nsim = sims_per_subj, paths = true, data = false)

    ### process the results
    ests = summarize_paths(paths_sim, model_sim.data)

    # get asymptotic bootstrap CIs
    asymp_cis = asymptotic_bootstrap(model_sim, model_fitted.parameters, model_fitted.vcov, sims_per_subj, nboot)    
    
    ### return results
    return DataFrame(simnum = simnum, cens = cens, nulleff = nulleff, var = string.(collect(keys(ests))), ests = collect(ests), lower = asymp_cis[:,1], upper = asymp_cis[:,2])
end

# wrapper for doing work
# function dowork(jobs, results)
#     while true
#         job = take!(jobs)
#         fn, args = job[1], job[2:end]
#         result = eval(fn)(args...)
#         put!(results, result)
#     end
# end
