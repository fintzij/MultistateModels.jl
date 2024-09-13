using ArraysOfArrays
using BSplineKit
using Chain
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using MultistateModels
using RCall
using StatsBase
using Random

# function to make the parameters
function makepars()
    parameters = (h12 = [log(1.25), log(1.5)],
                  h13 = [log(1.25), log(1)],
                  h23 = [log(1.25), log(2)])
    return parameters
end

# function to make the assessment times
# ntimes = 4 gives observations every three months
# ntimes = 12 gives monthly
function make_obstimes(ntimes)    
# observation times
    times = collect(range(0.0, 1.0, length = ntimes + 1))
    times[Not([begin, end])] .+= (rand(Beta(1.5, 1.5), length(times[Not([begin, end])])) .- 0.5) .* diff(times)[1]
    
    return times 
end

# function to set up the model
function setup_model(; make_pars, data = nothing, nsubj = 250, family = "wei", ntimes = 6, spknots = nothing)
    
    # create hazards
    if (family != "sp1") & (family != "sp2")
        h12 = Hazard(@formula(0 ~ 1), family, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), family, 1, 3)
        h23 = Hazard(@formula(0 ~ 1), family, 2, 3)

    elseif family == "sp1"

        knots12 = spknots[1]
        knots13 = spknots[2]
        knots23 = spknots[3]

        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 1, knots = knots12[Not([begin, end])], boundaryknots = knots12[[begin, end]], extrapolation = "flat")
        h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 1, knots = knots13[Not([begin, end])], boundaryknots = knots13[[begin, end]], extrapolation = "flat")
        h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 1, knots = knots23[Not([begin, end])], boundaryknots = knots23[[begin, end]], extrapolation = "flat")

    elseif family == "sp2"
        
        knots12 = spknots[1]
        knots13 = spknots[2]
        knots23 = spknots[3]

        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 3, knots = knots12[Not([begin, end])], boundaryknots = knots12[[begin, end]], extrapolation = "flat", monotone = 0)
        h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 3, knots = knots13[Not([begin, end])], boundaryknots = knots13[[begin, end]], extrapolation = "flat", monotone = 0)
        h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 3, knots = knots23[Not([begin, end])], boundaryknots = knots23[[begin, end]], extrapolation = "flat")
    end
    
    # data for simulation parameters
    if isnothing(data)
        visitdays = ntimes == 12 ? [make_obstimes(12) for i in 1:nsubj] : 
                (ntimes == 6) ? [getindex(make_obstimes(12), [1, 3, 5, 7, 9, 11, 13]) for i in 1:nsubj] : [getindex(make_obstimes(12), [1, 4, 7, 10, 13]) for i in 1:nsubj]
        data = DataFrame(id = repeat(collect(1:nsubj), inner = ntimes),
                    tstart = reduce(vcat, map(x -> x[Not(end)], visitdays)),
                    tstop = reduce(vcat, map(x -> x[Not(1)], visitdays)),
                    statefrom = fill(1, nsubj * ntimes),
                    stateto = fill(1, nsubj * ntimes),
                    obstype = fill(2, nsubj * ntimes))
    end

    # create model
    model = multistatemodel(h12, h13, h23; data = data)

    # set parameters
    if make_pars
        parameters = makepars()
        set_parameters!(model, parameters) 
    end

    # return model
    return model
end

# observe data
function observe_subjdat(path, model)

    # grab subject's data
    i = path.subj
    subj_dat_raw = model.data[model.subjectindices[i], :]

    # times and states
    obstimes = unique(sort([0.0; subj_dat_raw.tstop; path.times[findall(path.states .== 3)]]))
    
    # sort!(obstimes)
    obsinds  = searchsortedlast.(Ref(path.times), obstimes)
    obsstates = path.states[obsinds]

    # make dataset
    subjdat = DataFrame(id = path.subj,
                        tstart = obstimes[Not(end)],
                        tstop = obstimes[Not(1)],
                        statefrom = obsstates[Not(end)],
                        stateto = obsstates[Not(1)])

    # cull redundant rows in absorbing state
    subjdat = subjdat[Not((subjdat.stateto .== 3) .& (subjdat.statefrom .== 3)), :]

    # if subject passed through 2 on the way to 3 record it if not in the subject data
    if all([2,3] .∈ Ref(path.states)) && !(all([2,3] .∈ Ref(subjdat.stateto))) 
        # insert a ghost transition 
        insert!(subjdat, nrow(subjdat), 
                (id = i, tstart = subjdat.tstart[end], tstop = subjdat.tstop[end] - 0.0001, statefrom = 1, stateto = 2))

        subjdat.tstart[end]    = subjdat.tstop[end] - 0.0001
        subjdat.statefrom[end] = 2
    end

    # obstype
    subjdat[:,:obstype] .= [x == 3 ? 1 : 2 for x in subjdat[:,:stateto]]
    # subjdat[findall(subjdat.statefrom .== subjdat.stateto),:obstype] .= 1

    # return subjdat
    return subjdat
end

# summarize paths
function get_estimates(paths)

    # progression-free survival
    # percent who progress
    # percent who die after progression
    # percent who die without progressing
    pfs = mean(map(x -> all(x.states .== 1), paths))
    prog = mean(map(x -> 2 ∈ x.states, paths))
    die_wprog = mean(map(x -> all([2,3] .∈ Ref(x.states)), paths))
    die_noprog = mean(map(x -> (3 ∈ x.states) & !(2 ∈ x.states), paths))

    # restricted mean progression-free survival time
    rmpfst = mean(map(x -> x.times[2], paths))

    # time to disease among the progressors
    proginds = map(x -> (2 ∈ x.states) && (length(x.states) != 2), paths)
    time2prog = mapreduce(x -> x.times[2], +, paths[proginds]; init = 0.0) / sum(proginds)
    illnessdur = mapreduce(x -> (x.times[3] - x.times[2]), + , paths[proginds]; init = 0.0) / sum(proginds)

    # return estimates
    ests = (pfs = pfs, 
            prog = prog,
            die_wprog = die_wprog, 
            die_noprog = die_noprog, 
            rmpfst = rmpfst,
            time2prog = time2prog,
            illnessdur = illnessdur)

    return ests
end

# asymptotic bootstrap 
function asymptotic_bootstrap(model, pars, vcov, sims_per_subj, nboot)

    # draw parameters
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
    ests = zeros(Float64, 7, nboot)

    # simulate paths under each set of parameters
    for k in 1:nboot
        # draw parameters
        pardraws[1:npars] = flatview(pars) .+ S * randn(npars)

        # set the parameters
        set_parameters!(model, VectorOfVectors(pardraws, model.parameters.elem_ptr))

        # simulate paths
        paths_sim = simulate(model; nsim = sims_per_subj, paths = true, data = false)

        # summarize paths
        ests[:,k] = collect(get_estimates(paths_sim))
    end

    return mapslices(x -> quantile(skipmissing(x[findall(.!isnan.(x))]), [0.025, 0.975]), ests, dims = [2,])
end

# wrapper for one simulation
# nsim is the number of simulated paths per subject
# ndraws is the number of draws from the asymptotic normal distribution of the MLEs
# family:
#   1: "exp"
#   2: "wei"
#   3: "sp1" - degree 1 with knot at midpoint and range
#   4: "sp2" - degree 3 with knots at 0.05, 1/3 and 2/3, and 0.95 quantiles
function work_function(;simnum, seed, family, ntimes, sims_per_subj, nboot)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", ntimes = ntimes, nsubj = 250)
        
    # simulate paths
    paths = simulate(model_sim; nsim = 1, paths = true, data = false)
    dat = reduce(vcat, map(x -> observe_subjdat(x, model_sim), paths))

    if family < 3
        spknots = nothing
    elseif family == 3
        q12 = [0.0; quantile(MultistateModels.extract_sojourns(1, 2, MultistateModels.extract_paths(dat)), [0.5, 1.0])]
        q13 = [0.0; quantile(MultistateModels.extract_sojourns(1, 3, MultistateModels.extract_paths(dat)), [0.5, 1.0])]
        q23 = [0.0; quantile(MultistateModels.extract_sojourns(2, 3, MultistateModels.extract_paths(dat)), [0.5, 1.0])]        

        spknots = (knots12 = q12, knots13 = q13, knots23 = q23)
    elseif family == 4
        q12 = [0.0; quantile(MultistateModels.extract_sojourns(1, 2, MultistateModels.extract_paths(dat)), [1/3, 2/3, 1.0])]
        q13 = [0.0; quantile(MultistateModels.extract_sojourns(1, 3, MultistateModels.extract_paths(dat)), [1/3, 2/3, 1.0])]
        q23 = [0.0; quantile(MultistateModels.extract_sojourns(2, 3, MultistateModels.extract_paths(dat)), [1/3, 2/3, 1.0])]

        spknots = (knots12 = q12, knots13 = q13, knots23 = q23)
    end

    ### set up model for fitting
    model_fit = setup_model(; make_pars = false, data = dat, family = ["exp", "wei", "sp1", "sp2"][family], ntimes = ntimes, spknots = spknots)

    # fit model
    initialize_parameters!(model_fit)
    model_fitted = fit(model_fit; verbose = true, compute_vcov = true, ess_target_initial = 50, α = 0.2, γ = 0.2, tol = 0.001) 

    ### simulate from the fitted model
    model_sim2 = setup_model(; make_pars = false, data = model_sim.data, family = ["exp", "wei", "sp1", "sp2"][family], ntimes = ntimes, spknots = spknots)

    set_parameters!(model_sim2, model_fitted.parameters)
    paths_sim = simulate(model_sim2; nsim = sims_per_subj, paths = true, data = false)

    ### process the results
    ests = get_estimates(paths_sim)

    # get asymptotic bootstrap CIs
    asymp_cis = asymptotic_bootstrap(model_sim2, flatview(model_fitted.parameters), model_fitted.vcov,  sims_per_subj, nboot)    

    results = DataFrame(simnum = simnum, family = family, ntimes = ntimes, var = string.(collect(keys(ests))), ests = collect(ests), lower = asymp_cis[:,1], upper = asymp_cis[:,2])

    ### return results
    return results
end

# function for summarizing the crude estimates and paths
function summarize_crude(paths, dat, model)

    # get crude estimates for event probabilities
    nsubj = size(paths, 1)
    events = collect(get_estimates(paths))
    cis = rcopy(R"binom::binom.confint($(events[1:4])*$nsubj, $nsubj, method = 'wilson')[,c('lower', 'upper')]")

    # prepare dataset for restricted mean survival time
    times = map(p -> p.times[2], paths)
    statuses = map(p -> (p.states[2] == 1 ? 0.0 : 1.0), paths)
    
    # get restricted mean survival time
    rmst = rcopy(R"sm = survival:::survmean(survfit(Surv($times, $statuses) ~ 1), rmean = max($times))[[1]][c('rmean', 'se(rmean)')];c(est = sm[1], lower = sm[1] - 1.96 * sm[2], upper = sm[1] + 1.96 * sm[2])")

    # prepare dataset for time to progression
    proginds = map(x -> (2 ∈ x.states) && (length(x.states) != 2), paths)
    progtimes = map(x -> x.times[2], paths[proginds])
    illnessdurs = map(x -> (x.times[3] - x.times[2]), paths[proginds])
    
    # time to progression among progressors
    time2prog = rcopy(R"sm = survival:::survmean(survfit(Surv($progtimes, $(ones(length(progtimes)))) ~ 1), rmean = max($times))[[1]][c('rmean', 'se(rmean)')];c(est = sm[1], lower = sm[1] - 1.96 * sm[2], upper = sm[1] + 1.96 * sm[2])")

    # restricted mean illness duration among progressors
    illnessdur = rcopy(R"sm = survival:::survmean(survfit(Surv($illnessdurs, $(ones(length(illnessdurs)))) ~ 1), rmean = max($times))[[1]][c('rmean', 'se(rmean)')];c(est = sm[1], lower = sm[1] - 1.96 * sm[2], upper = sm[1] + 1.96 * sm[2])")

    ests_crude = DataFrame(ests = [events[1:4]; rmst[1]; time2prog[1]; illnessdur[1]],
                           lower = [cis.lower; rmst[2]; time2prog[2]; illnessdur[2]],
                           upper = [cis.upper; rmst[3]; time2prog[3]; illnessdur[3]])
end

# function for getting crude estimates
function crude_ests(;seed, ntimes)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", ntimes = ntimes, nsubj = 250)
        
    # simulate paths
    paths = simulate(model_sim; nsim = 1, paths = true, data = false)
    dat = reduce(vcat, map(x -> observe_subjdat(x, model_sim), paths))
    paths = MultistateModels.extract_paths(dat)

    # get estimates and confidence intervals
    ests_crude = summarize_crude(paths, dat, model_sim)

    return hcat(DataFrame(simnum = seed, 
                     ntimes = ntimes,
                     family = "crude",
                     var = ["pfs", "prog", "die_wprog", "die_noprog", "rmpfst", "time2prog", "illnessdur"]),
                     ests_crude)
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
