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
    parameters = (h12 = [log(1.5), log(1)],
                  h13 = [log(2/3), log(1)],
                  h23 = [log(1), log(1.25)])
    return parameters
end

# function to make the assessment times
function make_obstimes()    
    # observation times
    times = [0.0; collect(0.5:0.5:3.5) .+ (rand(Beta(2, 2), 7) .- 0.25) .* 0.25; 4.0]
    
    return times / 4.0
end

# function to set up the model
function setup_model(; make_pars, data = nothing, nsubj = 200, family = "wei", ntimes = 8, knots = nothing)
    
    # create hazards
    if (family != "sp1") & (family != "sp2")
        h12 = Hazard(@formula(0 ~ 1), family, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), family, 1, 3)
        h23 = Hazard(@formula(0 ~ 1), family, 2, 3)

    elseif family == "sp1"

        knots12 = knots[1]
        knots13 = knots[2]
        knots23 = knots[3]

        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 1, knots = knots12[2], boundaryknots = knots12[Not(2)], extrapolation = "flat")
        h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 1, knots = knots13[2], boundaryknots = knots13[Not(2)], extrapolation = "flat")
        h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 1, boundaryknots = [0.0, eps()], extrapolation = "flat")

    elseif family == "sp2"
        
        knots12 = knots[1]
        knots13 = knots[2]

        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 1, knots = knots12)
        h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 1, knots = knots13)
        h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 1, boundaryknots = [0.0, 1.0])
    end
    
    # data for simulation parameters
    if isnothing(data)
        visitdays = [make_obstimes() for i in 1:nsubj]
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

    # if path is 1->2->3 and !2 ∈ obsstates, include it
    if all([2,3] .∈ Ref(path.states)) && (last(path.times) < first(subj_dat_raw.tstop))
        push!(obstimes, path.times[findfirst(path.states .== 3)] - 0.0001)
    end
    
    sort!(obstimes)
    obsinds  = searchsortedlast.(Ref(path.times), obstimes)
    obsstates = path.states[obsinds]

    # make dataset
    subjdat = DataFrame(id = path.subj,
                        tstart = obstimes[Not(end)],
                        tstop = obstimes[Not(1)],
                        statefrom = obsstates[Not(end)],
                        stateto = obsstates[Not(1)])

    # cull redundatnt rows
    subjdat = subjdat[Not((subjdat.stateto .== 3) .& (subjdat.statefrom .== 3)), :]

    # obstype
    subjdat[:,:obstype] .= [x == 3 ? 1 : 2 for x in subjdat[:,:stateto]]

    # return subjdat
    return subjdat
end

# summarize paths
function summarize_paths(paths)

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

    ests = (pfs = pfs, 
            prog = prog,
            die_wprog = die_wprog, 
            die_noprog = die_noprog, 
            rmpfst = rmpfst)

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
    ests = zeros(Float64, 5, nboot)

    # simulate paths under each set of parameters
    for k in 1:nboot
        # draw parameters
        pardraws[1:npars] = flatview(pars) .+ S * randn(npars)

        # set the parameters
        set_parameters!(model, VectorOfVectors(pardraws, model.parameters.elem_ptr))

        # simulate paths
        paths_sim = simulate(model; nsim = sims_per_subj, paths = true, data = false)

        # summarize paths
        ests[:,k] = collect(summarize_paths(paths_sim))
    end

    return mapslices(x -> quantile(x[findall(map(y -> (!ismissing(y) && (y != -1.0)), x))], [0.025, 0.975]), ests, dims = [2,])
end

# wrapper for one simulation
# nsim is the number of simulated paths per subject
# ndraws is the number of draws from the asymptotic normal distribution of the MLEs
# family:
#   1: "exp"
#   2: "wei"
#   3: "sp1" - degree 1 with knot at midpoint and range
#   4: "sp2" - degree 3 with knots at 0.05, 1/3 and 2/3, and 0.95 quantiles
function work_function(;simnum, seed, family, sims_per_subj, nboot)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", nsubj = 200)
        
    # simulate paths
    paths = simulate(model_sim; nsim = 1, paths = true, data = false)
    dat = reduce(vcat, map(x -> observe_subjdat(x, model_sim), paths))

    if family < 3
        spknots = nothing
    elseif family == 3
        q12 = quantile(MultistateModels.extract_sojourns(1, 2, MultistateModels.extract_paths(dat; self_transitions = false)), [0.1, 0.5, 0.9])
        q13 = quantile(MultistateModels.extract_sojourns(1, 3, MultistateModels.extract_paths(dat; self_transitions = false)), [0.1, 0.5, 0.9])
        q23 = quantile(MultistateModels.extract_sojourns(2, 3, MultistateModels.extract_paths(dat; self_transitions = false)), [0.25, 0.75])

        spknots = (knots12 = q12, knots13 = q13, knots23 = q23)

    elseif family == 4
        spknots = (knots12 = quantile(MultistateModels.extract_sojourns(model_sim.hazards[1], dat, MultistateModels.extract_paths(dat; self_transitions = false)), [1/3, 2/3]),
        knots13 = quantile(MultistateModels.extract_sojourns(model_sim.hazards[2], dat, MultistateModels.extract_paths(dat; self_transitions = false)), [1/3, 2/3]))
    end

    ### set up model for fitting
    model_fit = setup_model(; make_pars = false, data = dat, family = ["exp", "wei", "sp1", "sp2"][family], knots = spknots)

    # fit model
    initialize_parameters!(model_fit)
    model_fitted = fit(model_fit; verbose = true, compute_vcov = true) 

    ### simulate from the fitted model
    spknots = family > 2 ? knots = (model_fitted.hazards[1].knots, model_fitted.hazards[2].knots) : nothing
    model_sim2 = setup_model(; make_pars = false, data = model_sim.data, family = ["exp", "wei", "sp1", "sp2"][family], knots = spknots)

    set_parameters!(model_sim2, model_fitted.parameters)
    paths_sim = simulate(model_sim2; nsim = sims_per_subj, paths = true, data = false)

    ### process the results
    ests = summarize_paths(paths_sim)

    # get asymptotic bootstrap CIs
    asymp_cis = asymptotic_bootstrap(model_sim2, flatview(model_fitted.parameters), model_fitted.vcov,  sims_per_subj, nboot)    

    results = DataFrame(simnum = simnum, family = family, var = string.(collect(keys(ests))), ests = collect(ests), lower = asymp_cis[:,1], upper = asymp_cis[:,2])

    ### return results
    return results
end

# function for summarizing the crude estimates and paths
function summarize_crude(paths, dat)

    # get crude estimates for event probabilities
    nsubj = size(paths, 1)
    probs = collect(summarize_paths(paths))[Not(end)]
    cis = rcopy(R"binom::binom.confint($probs*$nsubj, $nsubj, method = 'wilson')[,c('lower', 'upper')]")

    # prepare dataset for restricted mean survival time
    times = map(p -> p.times[2], paths)
    statuses = map(p -> (p.states[2] == 1 ? 0.0 : 1.0), paths)
    
    # get restricted mean survival time
    rmst = rcopy(R"sm = survival:::survmean(survfit(Surv($times, $statuses) ~ 1), rmean = max($times))[[1]][c('rmean', 'se(rmean)')];c(est = sm[1], lower = sm[1] - 1.96 * sm[2], upper = sm[1] + 1.96 * sm[2])")

    ests_crude = DataFrame(ests = [probs; rmst[1]],
                           lower = [cis.lower; rmst[2]],
                           upper = [cis.upper; rmst[3]])
end

# function for getting crude estimates
function crude_ests(;seed)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", nsubj = 1000)
        
    # simulate paths
    dat = simulate(model_sim; nsim = 1, paths = false, data = true)[1]
    paths = MultistateModels.extract_paths(dat)
    
    # get estimates and confidence intervals
    ests_crude = summarize_crude(paths, dat)

    return hcat(DataFrame(simnum = seed, 
                     family = "crude",
                     var = ["pfs", "prog", "die_wprog", "die_noprog","rmpfst"]),
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
