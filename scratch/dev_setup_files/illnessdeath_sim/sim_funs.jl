# function to make the parameters
function makepars()
    parameters = (h12 = [log(1.5), log(1.5)],
                  h13 = [log(2/3), log(2/3)],
                  h23 = [log(2), log(3)])
    return parameters
end

# function to make the assessment times
function make_obstimes()    
    # observation times
    times = collect(0:0.25:1) .+ vcat([0.0; (rand(Beta(5,5), 4) * 0.2 .- 0.1)])
    
    return times
end

# function to set up the model
function setup_model(; make_pars, data = nothing, nsubj = 1000, family = "wei", ntimes = 4)
    
    # create hazards
    if family != "sp"
        h12 = Hazard(@formula(0 ~ 1), family, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), family, 1, 3)
        h23 = Hazard(@formula(0 ~ 1), family, 2, 3)
    else
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree = 0)
        h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree = 0)
        h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree = 0)
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
    else
        set_crude_init!(model)
    end

    # return model
    return model
end

# calculate restricted mean progression free survival time
function calc_rmpfst(paths)
    # get event times
    died = map(x -> x.states[2] != 1, paths)
    gaptimes = diff([0.0; sort(map(x -> x.times[2], paths[findall(died)])); maximum(map(x -> x.times[2], paths))])

    # counters
    rmpfst = 0.0
    nsurv  = length(paths)

    for k in 1:(length(gaptimes)-1)
        rmpfst += (nsurv - 0.5) * gaptimes[k] 
        nsurv -= 1
    end

    # add final increment
    rmpfst += nsurv * last(gaptimes)

    # return restricted mean PFS time
    return rmpfst / length(paths)
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
    rmpfst = calc_rmpfst(paths)

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
#   3: "sp"
function work_function(;simnum, seed, family, sims_per_subj, nboot)

    Random.seed!(seed)

    # set up model for simulation
    model_sim = setup_model(; make_pars = true, data = nothing, family = "wei", nsubj = 1000)
        
    # simulate paths
    dat = simulate(model_sim; nsim = 1, paths = false, data = true)[1]

    ### set up model for fitting
    model_fit = setup_model(; make_pars = false, data = dat, family = ["exp", "wei", "sp"][family])

    # fit model
    model_fitted = fit(model_fit; verbose = true, compute_vcov = true) 

    ### simulate from the fitted model
    model_sim2 = setup_model(; make_pars = false, data = model_sim.data, family = ["exp", "wei", "sp"][family])

    set_parameters!(model_sim2, model_fitted.parameters)
    paths_sim = simulate(model_sim2; nsim = sims_per_subj, paths = true, data = false)

    ### process the results
    ests = summarize_paths(paths_sim)

    # get asymptotic bootstrap CIs
    asymp_cis = asymptotic_bootstrap(model_sim2, flatview(model_fitted.parameters), model_fitted.vcov,  sims_per_subj, nboot)    
    
    ### return results
    return DataFrame(simnum = simnum, family = family, var = string.(collect(keys(ests))), ests = collect(ests), lower = asymp_cis[:,1], upper = asymp_cis[:,2])
end

# function for summarizing the crude estimates and paths
function summarize_crude(paths, dat)

    # get crude estimates for event probabilities
    nsubj = size(paths, 1)
    probs = collect(summarize_paths(paths))[Not(end)]
    cis = rcopy(R"binom.confint($probs*$nsubj, $nsubj, method = 'wilson')[,c('lower', 'upper')]")

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
