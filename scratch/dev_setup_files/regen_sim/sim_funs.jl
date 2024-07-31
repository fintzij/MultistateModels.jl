# packages
using ArraysOfArrays
using Chain
using DataFrames
using DataFramesMeta
using Distributions
using LinearAlgebra
using MultistateModels
using StatsBase
using Random
using RCall

# function to make the parameters
function makepars(; model_number = 1, nulleff = 1)
    te = 2 - nulleff == 1

    if model_number <= 2
        parameters = (h12 = [log(0.7), log(0.6), log(0.33) * te],
                      h23 = [log(0.2), log(0.5) * te],
                      h24 = [log(0.5), log(1.4) * te],
                      h35 = [log(0.5), log(1.4) * te],
                      h45 = [log(0.3), log(0.5) * te],
                      h26 = [log(1.5), log(1), log(0.4) * te],
                      h67 = [log(1), log(0.5) * te],
                      h68 = [log(0.3), log(1.4) * te],
                      h79 = [log(0.3), log(1.4) * te],
                      h89 = [log(1), log(0.5) * te])
    else
        parameters = (h12 = [log(0.7), log(0.6), log(0.33) * te],
                      h23 = [log(0.5), log(1.4) * te],
                      h24 = [log(1.5), log(1), log(0.5) * te],
                      h45 = [log(0.3), log(1.4) * te])
    end
    return parameters
end

# function to make the PCR assessment times
function makepcrs(;ntimes=4)

    if ntimes == 4
        times = [0.0; [1.0, 2.0, 3.0] .+ (rand(Beta(3, 3), 3) .- 0.5) .* 0.5; 4.0]
    elseif ntimes == 1
        times = [0.0, 4.0]
    end
    
    return times
end

# remake dataset for prediction
function make_dat4pred(dat; obstype=2) 
    data = @chain dat begin
        groupby(:id)
        @combine(:tstart = 0.0,
                 :tstop = 4.0,
                 :statefrom = 1,
                 :stateto = 1,
                 :obstype = obstype,
                 :mab = first(:mab), 
                 :covid = maximum(:covid),
                 :sero = first(:sero))
    end
end

# function to set up the model
function setup_full_model(; make_pars, data = nothing, ntimes = 4, nulleff = 1, SamplingWeights = nothing, n_per_arm = 800)
    
    # create hazards
    h12 = Hazard(@formula(0 ~ 1 + mab), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1 + mab), "exp", 2, 3)
    h24 = Hazard(@formula(0 ~ 1 + mab), "exp", 2, 4)
    h35 = Hazard(@formula(0 ~ 1 + mab), "exp", 3, 5)
    h45 = Hazard(@formula(0 ~ 1 + mab), "exp", 4, 5)
    h26 = Hazard(@formula(0 ~ 1 + mab), "wei", 2, 6)
    h67 = Hazard(@formula(0 ~ 1 + mab), "exp", 6, 7)
    h68 = Hazard(@formula(0 ~ 1 + mab), "exp", 6, 8)
    h79 = Hazard(@formula(0 ~ 1 + mab), "exp", 7, 9)
    h89 = Hazard(@formula(0 ~ 1 + mab), "exp", 8, 9)

    # data for simulation parameters
    if isnothing(data)
        visitdays = [makepcrs(;ntimes = ntimes) for i in 1:(2*n_per_arm)]
        data = DataFrame(id = repeat(collect(1:(2 * n_per_arm )), inner = ntimes),
                    tstart = reduce(vcat, map(x -> x[Not(end)], visitdays)),
                    tstop = reduce(vcat, map(x -> x[Not(1)], visitdays)),
                    statefrom = fill(1, 2 * n_per_arm * ntimes),
                    stateto = fill(1, 2 * n_per_arm * ntimes),
                    obstype = fill(1, 2 * n_per_arm * ntimes),
                    mab = repeat([0.0, 1.0], inner = n_per_arm * ntimes),
                    covid = 0.0,
                    sero  = 0.0)
    end

    if isnothing(SamplingWeights)
        SamplingWeights = ones(Float64, length(unique(data.id)))
    end

    # create model
    model = multistatemodel(h12, h23, h24, h35, h45, h26, h67, h68, h79, h89; data = data, SamplingWeights = SamplingWeights)

    # set parameters
    if make_pars
        parameters = makepars(; model_number = 1, nulleff = nulleff)
        set_parameters!(model, parameters) 
    end

    # return model
    return model
end

# function to set up the model
function setup_collapsed_model(; make_pars, data, model_number, ntimes = 4, nulleff = 1, SamplingWeights = nothing)
    
    # create hazards
    if model_number == 3
        h12 = Hazard(@formula(0 ~ 1 + mab), "wei", 1, 2)
        h23 = Hazard(@formula(0 ~ 1 + mab), "exp", 2, 3)
        h24 = Hazard(@formula(0 ~ 1 + mab), "wei", 2, 4)
        h45 = Hazard(@formula(0 ~ 1 + mab), "exp", 4, 5)
    elseif model_number == 4
        h12 = Hazard(@formula(0 ~ 1 + mab), "sp", 1, 2, degree = 1, knots = [0.0, 5/7, 4.0])
        h23 = Hazard(@formula(0 ~ 1 + mab), "sp", 2, 3, degree = 1, knots = [0.0, 1.0, 4.0])
        h24 = Hazard(@formula(0 ~ 1 + mab), "sp", 2, 4, degree = 1, knots = [0.0, 1.0, 4.0])
        h45 = Hazard(@formula(0 ~ 1 + mab), "sp", 4, 5, degree = 1, knots = [0.0, 1.0, 4.0])        
    end

    if isnothing(SamplingWeights)
        SamplingWeights = ones(Float64, length(unique(data.id)))
    end
    
    # censoring patterns
    censoring_patterns = [3 1 0 1 0 0]
    
    # create model
    model = multistatemodel(h12, h23, h24, h45; data = data, SamplingWeights = SamplingWeights, CensoringPatterns = censoring_patterns)

    # set parameters
    if make_pars
        parameters = makepars(; model_number = 3, nulleff = nulleff)
        set_parameters!(model, parameters) 
    end

    # return model
    return model
end

# function to collapse path from 9 states to 5
function collapse_path(path_full)
    times  = path_full.times
    states = path_full.states

    # recode
    states[findall(states .== 1)]         .= -1
    states[findall(states .∈ Ref([2,3]))] .= -2
    states[findall(states .∈ Ref([4,5]))] .= -3
    states[findall(states .∈ Ref([6,7]))] .= -4
    states[findall(states .∈ Ref([8,9]))] .= -5

    states .*= -1

    # indices
    inds = [0; findall(states[Not(1)] .!= states[Not(end)])] .+ 1

    # tack on last state and time if not absorbing
    if (last(states, 2) == [1,1]) | (last(states, 2) == [2,2]) | (last(states, 2) == [4,4])
        push!(inds, length(states))
    end

    # remove duplicates
    states = states[inds]
    times  = times[inds]

    return MultistateModels.SamplePath(path_full.subj, times, states)
end

# function to wrangle a path to subject data
function observe_subjdat_full(path, model)
    # grab subject's data
    i = path.subj
    subj_dat_raw = model.data[model.subjectindices[i], :]

    # make dataset
    ntimes  = length(path.times) - 1 
    subjdat = DataFrame(
        id = fill(path.subj, ntimes),
        tstart = path.times[Not(end)],
        tstop = path.times[Not(1)],
        statefrom = path.states[Not(end)],
        stateto = path.states[Not(1)],
        obstype = fill(1, ntimes),
        mab = fill(subj_dat_raw.mab[1], ntimes))
    
    # tack on PCR, symptoms, sero
    subjdat[:,:pcrpos] = ifelse.(subjdat.stateto .∈ Ref([2,3,6,7]), 1, 0)
    subjdat[:,:covid] = ifelse.(subjdat.stateto .> 5, 1, 0)
    subjdat[:,:sero] = ifelse.(subjdat.stateto .∈ Ref([3,5,7,9]), 1, 0)

    # cull out redundant observations in absorbing states
    subjdat = subjdat[Not(((subjdat.stateto .== 5) .& (subjdat.statefrom .== 5)) .| ((subjdat.stateto .== 9) .& (subjdat.statefrom .== 9))), :]

    # exit
    return subjdat
end

function observe_subjdat_panel(path, model)

    # grab subject's data
    i = path.subj
    subj_dat_raw = model.data[model.subjectindices[i], :]

    # sequences of times and states
    obstimes = unique(sort([0.0; subj_dat_raw.tstop; path.times[findall(path.states .== 6)]]))
    obsinds  = searchsortedlast.(Ref(path.times), obstimes)
    obsstates = path.states[obsinds]

    # if there is a 1->6 observation insert a ghost times
    if any((obsstates[Not(end)] .== 1) .& (obsstates[Not(1)] .== 6))
        # insert time and state
        push!(obstimes, path.times[findfirst(path.states .== 6)] - sqrt(eps()))
        push!(obsstates, 2)

        # reorder
        ord = sortperm(obstimes)
        obstimes = obstimes[ord]
        obsstates = obsstates[ord]

        # make sure there is no delta less than sqrt(eps())
        if any(diff(obstimes) .< sqrt(eps()))
            obstimes .+= collect(0:(length(obstimes)-1)) * sqrt(eps())
        end
    end
    
    # make data
    ntimes  = length(obstimes) - 1 
    subjdat = DataFrame(
        id = fill(path.subj, ntimes),
        tstart = obstimes[Not(end)],
        tstop = obstimes[Not(1)],
        statefrom = obsstates[Not(end)],
        stateto = obsstates[Not(1)],
        obstype = fill(2, ntimes),
        mab = fill(subj_dat_raw.mab[1], ntimes))

    # tack on PCR, symptoms, sero
    subjdat[:,:pcr] = ifelse.(subjdat.stateto .∈ Ref([2,3,6,7]), 1, 0)
    subjdat[:,:covid] = ifelse.(subjdat.stateto .> 5, 1, 0)
    subjdat[:,:sero] = ifelse.(subjdat.stateto .∈ Ref([3,5,7,9]), 1, 0)

    # correct obstype
    if any(subjdat.stateto .== 6)
        subjdat.obstype[findfirst(subjdat.stateto .== 6)] = 1
    end

    # cull out redundant observations in absorbing states
    subjdat = subjdat[Not(((subjdat.stateto .== 5) .& (subjdat.statefrom .== 5)) .| ((subjdat.stateto .== 9) .& (subjdat.statefrom .== 9))), :]

    # exit
    return subjdat
end

function observe_subjdat_collapsed(path, model)

    # grab subject's data
    i = path.subj
    subj_dat_raw = model.data[model.subjectindices[i], :]

    # get serology 
    sero = any(path.states .∈ Ref([3,5,7,9]))

    # collapse path
    path = collapse_path(deepcopy(path))

    # sequences of times and states
    obstimes = unique(sort([0.0; subj_dat_raw.tstop; path.times[findall(path.states .== 4)]]))
    obsinds  = searchsortedlast.(Ref(path.times), obstimes)
    obsstates = path.states[obsinds]

    # if there is a 1->6 observation insert a ghost times
    if any((obsstates[Not(end)] .== 1) .& (obsstates[Not(1)] .== 4))
        # insert time and state
        push!(obstimes, path.times[findfirst(path.states .== 4)] - sqrt(eps()))
        push!(obsstates, 2)

        # reorder
        ord = sortperm(obstimes)
        obstimes = obstimes[ord]
        obsstates = obsstates[ord]

        # make sure there is no delta less than sqrt(eps())
        if any(diff(obstimes) .< sqrt(eps()))
            obstimes .+= collect(0:(length(obstimes)-1)) * sqrt(eps())
        end
    end
    
    # make data
    ntimes  = length(obstimes) - 1 
    subjdat = DataFrame(
        id = fill(path.subj, ntimes),
        tstart = obstimes[Not(end)],
        tstop = obstimes[Not(1)],
        statefrom = obsstates[Not(end)],
        stateto = obsstates[Not(1)],
        obstype = fill(2, ntimes),
        mab = fill(subj_dat_raw.mab[1], ntimes))

    # tack on PCR, symptoms, sero
    subjdat[:,:pcr]   = ifelse.(subjdat.stateto .∈ Ref([2,4]), 1, 0)
    subjdat[:,:covid] = ifelse.(subjdat.stateto .> 3, 1, 0)
    subjdat[:,:sero] .= sero
    
    # correct obstype
    if any(subjdat.stateto .== 4)
        subjdat.obstype[findfirst(subjdat.stateto .== 4)] = 1
    elseif sero & !any(subjdat.covid .== 1) & !any(subjdat.pcr .== 1)
        subjdat.stateto[Not(end)] .= 0
        subjdat.stateto[end]       = 3
        subjdat.statefrom[Not(1)] .= 0
        subjdat.obstype[Not(end)] .= 3
    end
    
    # cull out redundant observations in absorbing states
    subjdat = subjdat[Not(((subjdat.stateto .== 3) .& (subjdat.statefrom .== 3)) .| ((subjdat.stateto .== 5) .& (subjdat.statefrom .== 5))), :] 

    # exit
    return subjdat
end

# wrapper
function observe_subjdat(path, model, model_number)
    if model_number == 1
        subjdat = observe_subjdat_full(path, model)
    elseif model_number == 2
        subjdat = observe_subjdat_panel(path, model)
    else
        subjdat = observe_subjdat_collapsed(path, model)
    end
    return subjdat
end

# to determine if ever PCR+
function pcrdetec(path; full = true)
    
    # code events 
    covid = full ? 6 : 4
    pcrpos = full ? [2,3,6,7] : [2,4]
    
    # get the observation times
    obsstates = path.states[searchsortedlast.(Ref(path.times), collect(0.0:4.0))]
    
    # return 0 or 1 if detected
    detected = any(obsstates .∈ Ref(pcrpos)) 

    return detected
end

# to determine if an infection is detected
function pcrdur_full(paths_pred, dat)

    # identify paths with infections
    infected = map(x -> x.states[2] > 1, paths_pred)

    # get durations
    pcrstart = map(x -> x.states[2] > 1 ? x.times[2] : 0.0, paths_pred)

    pcrstop = map(x -> 
        x.states[2] == 1 ? 0.0 : 
        last(x.states) ∈ [2,3,6,7] ? last(x.times) : 
        x.times[findfirst(x.states .∈ Ref([4,5,8,9]))], paths_pred)

    pcrdur = pcrstop .- pcrstart

    # compute the average duration among the infected
    durs = map((x,y) -> any(y) ? mean(x[y]) : missing, eachrow(pcrdur), eachrow(infected))

    return durs
end

# to calculate the probability of detection
function pcr_detectprob(paths, full)

    # find infected
    infected = map(x -> x.states[2] > 1, paths)

    # observe subjdat
    detected = pcrdetec.(paths; full = full)

    # probability of detection given infection
    probdetec = map((x,y) -> sum(y) == 0 ? missing : sum(x[findall(y .== 1)]) / sum(y), eachrow(detected), eachrow(infected))

    return probdetec
end

function summarize_paths_full(paths, model; return_counts = false)

    # get data object
    dat = make_dat4pred(model.data)

    # tabulate for npos
    dat[:,:weight]       = model.SamplingWeights
    dat[:,:infected]     = [mean(map(x -> any(x.states .> 1), y)) for y in eachrow(paths)]
    dat[:,:symptomatic]  = [mean(map(x -> any(x.states .> 5), y)) for y in eachrow(paths)]
    dat[:,:asymptomatic] = dat.infected .- dat.symptomatic
    dat[:,:seropos]      = [mean(map(x -> any(x.states .∈ Ref([3,5,7,9])), y)) for y in eachrow(paths)]
    dat[:,:serosym]      = [mean(map(x -> any(x.states .∈ Ref([7,9])), y)) for y in eachrow(paths)]
    dat[:,:seroasym]     = [mean(map(x -> any(x.states .∈ Ref([3,5])), y)) for y in eachrow(paths)]
    dat[:,:infectime]    = [mean(map(x -> any(x.states .> 1) ? x.times[findfirst(x.states .> 1)] : 4.0, y)) for y in eachrow(paths)]

    # get probability of detection|infection and pcr+ duration
    dat[:,:pcrdur] = pcrdur_full(paths, dat)
    dat[:,:pcrdetect] = pcr_detectprob(paths, true)
    
    # summarize by mab
    counts = @chain dat begin
        groupby(:mab)
        @combine(:N = sum(:weight),
                 :i = sum(:infected .* :weight),
                 :s = sum(:symptomatic .* :weight),
                 :a = sum(:asymptomatic .* :weight),
                 :sero = sum(:seropos .* :weight),
                 :serosym = sum(:serosym .* :weight),
                 :seroasym = sum(:seroasym .* :weight),
                 :rmt_i = sum(:infectime .* :weight) / sum(:weight),
                 :pcrdur = sum(:pcrdur[.!ismissing.(:pcrdur)] .* :weight[.!ismissing.(:pcrdur)]) / sum(:weight[.!ismissing.(:pcrdur)]),
                 :pcrdetect = sum(:pcrdetect[.!ismissing.(:pcrdetect)] .* :weight[.!ismissing.(:pcrdetect)]) / sum(:weight[.!ismissing.(:pcrdetect)]))
    end

    # collect estimates
    ests_wide = @chain counts begin
        groupby(:mab)
        @combine(:i = :i ./ :N,
                 :s = :s ./ :N,
                 :a = :a ./ :N,
                 :s_i = :s ./ :i,
                 :a_i = :a ./ :i,
                 :n_i = :sero ./ :i,
                 :n_s = :serosym ./ :s,
                 :n_a = :seroasym ./ :a,
                 :rmti = :rmt_i,
                 :pcrdur = :pcrdur,
                 :pcrdetect = :pcrdetect)
    end

    # stack the estimates
    ests = @chain ests_wide begin
        stack(Not(:mab))
        @transform(@byrow :var = :variable * (:mab == 0.0 ? "_p" : "_m"))
        @select($(Not(:mab, :variable)))
        @select(:var, :value)
    end 

    # add in contrasts
    append!(ests, DataFrame(
            var = ["pe_infec", 
                   "pe_sym", 
                   "pe_asym", 
                   "eff_sym_infec", 
                   "eff_n_infec", 
                   "eff_n_sym", 
                   "eff_n_asym"],
            value = [1 .- ests.value[ests.var .== "i_m"] ./ ests.value[ests.var .== "i_p"];
                     1 .- ests.value[ests.var .== "s_m"] ./ ests.value[ests.var .== "s_p"];
                     1 .- ests.value[ests.var .== "a_m"] ./ ests.value[ests.var .== "a_p"];
                     ests.value[ests.var .== "s_i_m"] ./ ests.value[ests.var .== "s_i_p"];
                     ests.value[ests.var .== "n_i_m"] ./ ests.value[ests.var .== "n_i_p"];
                     ests.value[ests.var .== "n_s_m"] ./ ests.value[ests.var .== "n_s_p"];
                     ests.value[ests.var .== "n_a_m"] ./ ests.value[ests.var .== "n_a_p"]]))

    if return_counts
        return counts
    else
        return ests
    end
end

# asymptotic bootstrap 
function asymptotic_bootstrap_full(model, pars, sigma, sims_per_subj, nboot)

    # draw parameters
    npars = length(pars)

    # SVD
    U = zeros(npars, npars) 
    D = zeros(npars)
    U,D = svd(sigma)

    # replace small negative singular values with zeros
    D[findall(D .< 0)] .= 0.0

    # matrix square root
    S = U * diagm(sqrt.(D))
    
    # initialize matrix of estimates    
    ests = zeros(Float64, 29, nboot)

    # simulate paths under each set of parameters
    for k in 1:nboot
        # set the parameters
        set_parameters!(model, VectorOfVectors(pars .+ S * randn(npars), model.parameters.elem_ptr))

        # simulate paths
        paths_sim = simulate(model; nsim = sims_per_subj, paths = true, data = false)

        # summarize paths
        ests[:,k] = reduce(vcat, collect(summarize_paths_full(paths_sim, model).value))
    end

    return mapslices(x -> [quantile(skipmissing(x), [0.025, 0.975]); std(skipmissing(x))], ests, dims = [2,])
end

# to determine if an infection is detected
function pcrdur_collapsed(paths_pred)

    # identify paths with infections
    infected = map(x -> x.states[2] > 1, paths_pred)

    # get durations
    pcrstart = map(x -> x.states[2] > 1 ? x.times[2] : 0.0, paths_pred)

    pcrstop = map(x -> 
        x.states[2] == 1 ? 0.0 : 
        last(x.states) ∈ [2,4] ? last(x.times) : 
        x.times[findfirst(x.states .∈ Ref([3,5]))], paths_pred)

    pcrdur = pcrstop .- pcrstart

    # compute the average duration among the infected
    durs = map((x,y) -> any(y) ? mean(x[y]) : missing, eachrow(pcrdur), eachrow(infected))

    return durs
end

# summarize paths
function summarize_paths_collapsed(paths_pred, model_pred; return_counts = false)

    # initialize data
    dat = deepcopy(model_pred.data)

    # summarize stateto-statefrom data at the individual level
    gdat = groupby(dat, :id)
    dat = combine(gdat,
        :covid => (x -> any(x.==1)) => :covid,
        :sero => (x -> any(x.==1)) => :sero,
        :mab => (x -> any(x.==1)) => :mab
    )

    # matrices with events
    infected = [any(x.states .> 1) for x in paths_pred]
    symptomatic = [any(x.states .> 3) for x in paths_pred]
    asymptomatic = infected .- symptomatic
    infectime = [any(x.states .> 1) ? x.times[findfirst(x.states .> 1)] : 4.0 for x in paths_pred]

    # observed in the data - not derived from simulated paths
    dat[:,:serosympt]  = dat[:,:sero] .* dat[:,:covid]
    dat[:,:seroasympt] = dat[:,:sero] .- dat[:,:serosympt]

    # tabulate for npos
    dat[:,:weight]       = model_pred.SamplingWeights
    dat[:,:infected]    .= mean(infected, dims = 2)
    dat[:,:symptomatic] .= mean(symptomatic, dims = 2)
    dat[:,:asymptomatic] = dat.infected .- dat.symptomatic
    dat[:,:infectime]   .= mean(infectime, dims = 2)

    # get probability of detection|infection and pcr+ duration
    dat[:,:pcrdur] = pcrdur_collapsed(paths_pred)
    dat[:,:pcrdetect] = pcr_detectprob(paths_pred, false)

     # summarize by mab
    counts = @chain dat begin
        groupby(:mab)
        @combine(:N = sum(:weight),
                # model estimates 
                 :i = sum(:infected .* :weight),
                 :s = sum(:symptomatic .* :weight),
                 :a = sum(:asymptomatic .* :weight),
                 # from data
                 :sero = sum(:sero .* :weight),
                 :serosym = sum(:serosympt .* :weight),
                 :seroasym = sum(:seroasympt .* :weight),
                 # model estimates
                 :rmt_i = sum(:infectime .* :weight) / sum(:weight),
                 :pcrdur = sum(:pcrdur[.!ismissing.(:pcrdur)] .* :weight[.!ismissing.(:pcrdur)]) / sum(:weight[.!ismissing.(:pcrdur)]),
                 :pcrdetect = sum(:pcrdetect[.!ismissing.(:pcrdetect)] .* :weight[.!ismissing.(:pcrdetect)]) / sum(:weight[.!ismissing.(:pcrdetect)]))
    end

    # collect estimates
    ests_wide = @chain counts begin
        groupby(:mab)
        @combine(:i = :i ./ :N,
                 :s = :s ./ :N,
                 :a = :a ./ :N,
                 :s_i = :s ./ :i,
                 :a_i = :a ./ :i,
                 :n_i = :sero ./ :i,
                 :n_s = :serosym ./ :s,
                 :n_a = :seroasym ./ :a,
                 :rmti = :rmt_i,
                 :pcrdur = :pcrdur,
                 :pcrdetect = :pcrdetect)
    end

    # stack the estimates
    ests = @chain ests_wide begin
        stack(Not(:mab))
        @transform(@byrow :var = :variable * (:mab == 1.0 ? "_m" : "_p"))
        @select($(Not(:mab, :variable)))
        @select(:var, :value)
    end 

    # add in contrasts
    append!(ests, DataFrame(
            var = ["pe_infec", 
                   "pe_sym", 
                   "pe_asym", 
                   "eff_sym_infec", 
                   "eff_n_infec", 
                   "eff_n_sym", 
                   "eff_n_asym"],
            value = [1 .- ests.value[ests.var .== "i_m"] ./ ests.value[ests.var .== "i_p"];
                     1 .- ests.value[ests.var .== "s_m"] ./ ests.value[ests.var .== "s_p"];
                     1 .- ests.value[ests.var .== "a_m"] ./ ests.value[ests.var .== "a_p"];
                     ests.value[ests.var .== "s_i_m"] ./ ests.value[ests.var .== "s_i_p"];
                     ests.value[ests.var .== "n_i_m"] ./ ests.value[ests.var .== "n_i_p"];
                     ests.value[ests.var .== "n_s_m"] ./ ests.value[ests.var .== "n_s_p"];
                     ests.value[ests.var .== "n_a_m"] ./ ests.value[ests.var .== "n_a_p"]]))

    if return_counts
        return counts
    else
        return ests
    end
end

# to get estimates from the collapsed model
function get_estimates_collapsed(model_fitted; sims_per_subj, model_number)

    # models for prediction
    model_pred = setup_collapsed_model(; make_pars = false, data = make_dat4pred(model_fitted.data), SamplingWeights = model_fitted.SamplingWeights, model_number = model_number)

    # set parameters
    set_parameters!(model_pred, model_fitted.parameters)
   
    # simulate paths
    paths_pred = simulate(model_pred; nsim = sims_per_subj, paths = true, data = false)

    # summarize paths
    ests = summarize_paths_collapsed(paths_pred, model_pred)

    # exit
    return ests
end

# wrapper to fit the collapsed model to npos or nneg
function fit_collapsed(dat; model_number, SamplingWeights = nothing)

    # fit to npos and nneg
    model_fit = setup_collapsed_model(; make_pars = false, data = dat, SamplingWeights = SamplingWeights, model_number = model_number)

    # initialize model parameters
    initialize_parameters!(model_fit; crude = true)
    initialize_parameters!(model_fit)

    # fit models
    model_fitted = fit(model_fit; verbose = true, compute_vcov = false, maxiter = 500, ess_target_initial = 100)

    return model_fitted
end

# wrapper for one simulation
function work_function(;seed1, seed2, model_number, nulleff)

    Random.seed!(seed1)

    # set up model for simulation
    model_full_sim = setup_full_model(; make_pars = true, data = nothing, nulleff = nulleff)
        
    # simulate paths
    paths = simulate(model_full_sim; nsim = 1, paths = true, data = false)[:,1]

    # make dataset
    dat = reduce(vcat, map(x -> observe_subjdat(x, model_full_sim, model_number), paths))

    # set second seed
    Random.seed!(seed2)

    ### set up model for fitting
    if model_number <= 2

        sims_per_subj = 20; nboot = 1000
        
        # remake model for fitting
        model_fit = setup_full_model(; make_pars = false, data = dat)

        # initialize model parameters
        initialize_parameters!(model_fit; crude = true)

        if model_number == 1
            model_fitted = fit(model_fit; verbose = true, compute_vcov = true) 
        elseif model_number == 2
            initialize_parameters!(model_fit)
            model_fitted = fit(model_fit; verbose = true, compute_vcov = true, maxiter = 500, ess_target_initial = 200)
        end

        # model for simulation
        model_pred = setup_full_model(; make_pars = false, data = make_dat4pred(model_fitted.data))
        set_parameters!(model_pred, model_fitted.parameters)

        # predict
        paths_pred = simulate(model_pred; nsim = sims_per_subj, paths = true, data = false)

        ### process the results
        ests = summarize_paths_full(paths_pred, model_pred)

        # get asymptotic bootstrap CIs
        cis = asymptotic_bootstrap_full(model_pred, flatview(model_fitted.parameters), model_fitted.vcov, sims_per_subj, nboot)    
        
        ### return results
        return DataFrame(seed1 = seed1, seed2 = seed2, model_number = model_number, nulleff = nulleff, var = ests.var, ests = ests.value, lower = cis[:,1], upper = cis[:,2], se = cis[:,3])

    else     
        # set sims_per_subj
        sims_per_subj = 100

        # subject weights
        if seed2 == 0
            # initialize bookkeeping for weights
            subjweights = @chain dat begin
                groupby(:id)
                @combine(:mab = first(:mab),
                        :sero = first(:sero),
                        :weight = 1.0)
            end
        else
            # initialize bookkeeping for weights
            subjweights = @chain dat begin
                groupby(:id)
                @combine(:mab = first(:mab),
                        :sero = first(:sero))
                groupby(:mab)
                @transform(:weight = length(:id) * vec(rand(Dirichlet(ones(length(:id))),1)))
            end
        end

        # fit models
        model_fitted = fit_collapsed(dat; model_number = model_number, SamplingWeights = subjweights.weight)
        
        # get estimates
        ests = get_estimates_collapsed(model_fitted; sims_per_subj = sims_per_subj, model_number = model_number)
        
        ### return results
        return DataFrame(simnum = seed1, iter = seed2, model_number = model_number, var = ests.var, est = ests.value)
    end     
end

# calculate the crude estimates
function summarize_crude(dat; give_counts = false)

    counts = @chain copy(dat) begin
        groupby(:id)
        @combine($AsTable = (
            mab = first(:mab),
            infected = any(:pcr .== 1) | any(:covid .== 1) | any(:sero .== 1),
            asymptomatic = (any(:pcr .== 1) | any(:sero .== 1)) & !any(:covid .== 1),
            symptomatic = any(:covid .== 1),
            npos = any(:sero .== 1),
            npos_asym = any(:sero .== 1) & !any(:covid .== 1),
            npos_sym = any(:sero .== 1) & any(:covid .== 1)))
        groupby(:mab)
        @combine($AsTable = (n = length(:id),
                    n_infected = sum(:infected),
                    n_asymptomatic = sum(:asymptomatic),
                    n_symptomatic = sum(:symptomatic),
                    n_npos = sum(:npos),
                    n_npos_asym = sum(:npos_asym),
                    n_npos_sym = sum(:npos_sym)))
    end

    if give_counts
        return counts
    else
        # initialize data frame
        ests = DataFrame()

        # infection prob
        append!(ests,
            hcat(["Pr(Infec. | Plac.)", "Pr(Infec. | mAb)"], rcopy(R"binom::binom.confint($(counts.n_infected), $(counts.n), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # symptomatic infection
        append!(ests,
            hcat(["Pr(Sympt. | Plac.)", "Pr(Sympt. | mAb)"], rcopy(R"binom::binom.confint($(counts.n_symptomatic), $(counts.n), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # asymptomatic infection
        append!(ests,
            hcat(["Pr(Asympt. | Plac.)", "Pr(Asympt. | mAb)"], rcopy(R"binom::binom.confint($(counts.n_asymptomatic), $(counts.n), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # symptoms | infection
        append!(ests,
            hcat(["Pr(Sympt. | Infec., Plac.)", "Pr(Sympt. | Infec., mAb)"], rcopy(R"binom::binom.confint($(counts.n_symptomatic), $(counts.n_infected), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # no symptoms | infection
        append!(ests,
            hcat(["Pr(Asympt. | Infec., Plac.)", "Pr(Asympt. | Infec., mAb)"], rcopy(R"binom::binom.confint($(counts.n_asymptomatic), $(counts.n_infected), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # n+ | infec
        append!(ests,
            hcat(["Pr(N+ | Infec., Plac.)", "Pr(N+ | Infec., mAb)"], rcopy(R"binom::binom.confint($(counts.n_npos), $(counts.n_infected), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # n+ | sympt
        append!(ests,
            hcat(["Pr(N+ | sympt., Plac.)", "Pr(N+ | sympt., mAb)"], rcopy(R"binom::binom.confint($(counts.n_npos_sym), $(counts.n_symptomatic), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # n+ | asympt
        append!(ests,
            hcat(["Pr(N+ | asympt., Plac.)", "Pr(N+ | asympt., mAb)"], rcopy(R"binom::binom.confint($(counts.n_npos_asym), $(counts.n_asymptomatic), method = 'wilson')[,c('mean', 'lower', 'upper')]")))

        # PE - resume here
        push!(ests,
            ["PE for infec."; rcopy(R"1 - DescTools::BinomRatioCI($(counts.n_infected[2]), $(counts.n[2]), $(counts.n_infected[1]), $(counts.n[1]), method = 'koopman')[c(1,3,2)]")])

        push!(ests,
            ["PE for sympt. infec."; rcopy(R"1 - DescTools::BinomRatioCI($(counts.n_symptomatic[2]), $(counts.n[2]), $(counts.n_symptomatic[1]), $(counts.n[1]), method = 'koopman')[c(1,3,2)]")])

        push!(ests,
            ["PE for asympt. infec."; rcopy(R"1 - DescTools::BinomRatioCI($(counts.n_asymptomatic[2]), $(counts.n[2]), $(counts.n_asymptomatic[1]), $(counts.n[1]), method = 'koopman')[c(1,3,2)]")])

        # symptoms given infection
        push!(ests,
            hcat("RR for sympt. | infec.", rcopy(R"DescTools::BinomRatioCI($(counts.n_symptomatic[2]), $(counts.n_infected[2]), $(counts.n_symptomatic[1]), $(counts.n_infected[1]), method = 'koopman')")))

        # seroconv
        push!(ests,
            hcat("RR for N+ | infec.", rcopy(R"DescTools::BinomRatioCI($(counts.n_npos[2]), $(counts.n_infected[2]), $(counts.n_npos[1]), $(counts.n_infected[1]), method = 'koopman')")))

        push!(ests,
            hcat("RR for N+ | sympt.", rcopy(R"DescTools::BinomRatioCI($(counts.n_npos_sym[2]), $(counts.n_symptomatic[2]), $(counts.n_npos_sym[1]), $(counts.n_symptomatic[1]), method = 'koopman')")))

        push!(ests,
            hcat("RR for N+ | asympt.", rcopy(R"DescTools::BinomRatioCI($(counts.n_npos_asym[2]), $(counts.n_asymptomatic[2]), $(counts.n_npos_asym[1]), $(counts.n_asymptomatic[1]), method = 'koopman')")))

        # rename
        rename!(ests, [:var, :est, :lower, :upper])

        # return
        return ests
    end
end

# function for getting crude estimates
function crude_ests(;seed, nulleff)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_full_model(; make_pars = true, data = nothing, nulleff = nulleff)
        
    # simulate paths
    paths = simulate(model_sim; nsim = 1, paths = true, data = false)[:,1]

    # make dataset
    dat = reduce(vcat, map(x -> observe_subjdat_collapsed(x, model_sim), paths))
    
    # get estimates and confidence intervals
    crude = summarize_crude(dat)

    # add bookkeeping
    crude[!,:seed] .= seed
    crude[!,:nulleff] .= nulleff
    crude[!,:cens] .= 3

    # join and reorder
    select!(crude, :seed, :nulleff, :cens, :)

    # return
    return crude
end
