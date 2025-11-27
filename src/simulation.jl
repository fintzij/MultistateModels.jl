# simulation functions
using Random
# simone_path: simulates a single sample path
# sim_paths: wrapper around simeone_path to simulate nsamples x nids number of paths
# simulate: wrapper around sim_paths and maybe other stuff to simulate new data, also incorporates censoring, etc.

"""
    simulate(model::MultistateProcess; nsim = 1, data = true, paths = false)

Simulate `n` datasets or collections of sample paths from a multistate model. If `data = true` (the default) discretely observed sample paths are returned, possibly subject to measurement error. If `paths = false` (the default), continuous-time sample paths are not returned.

# Arguments
- `model::MultistateProcess`: object created by multistatemodel()
- `nsim`: number of sample paths to simulate
- `data`: boolean; if true then return discretely observed sample paths
- `paths`: boolean; if false then continuous-time sample paths not returned
- `delta_u`: minimum cumulative incidence increment per-jump, defaults to the larger of (1 / #subjects)^2 or sqrt(eps())
- `delta_t`: minimum time increment per-jump, defaults to sqrt(eps())
 - `time_transform`: toggle Tang-style shared-trajectory solver. When `true` (default) the simulator reuses the same caches used by the exact-path likelihood; set to `false` to force the legacy cumulative-hazard path even for Tang-enabled hazards.
"""
function simulate(model::MultistateProcess; nsim = 1, data = true, paths = false, delta_u = sqrt(eps()), delta_t = sqrt(eps()), time_transform::Bool = true)

    # throw an error if neither paths nor data are asked for
    if paths == false && data == false
        error("Why are you calling `simulate` if you don't want sample paths or data? Stop wasting my time.")
    end

    # number of subjects
    nsubj = length(model.subjectindices)

    # initialize array for simulated paths 
    if paths == true
        samplepaths = Array{SamplePath}(undef, nsubj, nsim)
    end

    # initialize container for simulated data  
    if data == true
        datasets = Array{DataFrame}(undef, nsubj, nsim)
    end 

    for i in Base.OneTo(nsim)
        for j in Base.OneTo(nsubj)
            
            # simulate a path for subject j
            samplepath = simulate_path(model, j, delta_u, delta_t; time_transform = time_transform)

            # save path if requested
            if paths == true
                samplepaths[j, i] = samplepath
            end

            # observe path 
            if data == true
                datasets[j, i] = observe_path(samplepath, model)
            end
        end
    end

    # vertically concatenate datasets
    if data == true
        dat = mapslices(x -> reduce(vcat, x), datasets, dims = [1,])
    end

    # concatenate subject paths
    if paths == true
        trajectories = mapslices(x -> reduce(vcat, x), samplepaths, dims = [1,])
    end

    # return paths and data
    if paths == false && data == true
        return dat
    elseif paths == true && data == true
        return dat, trajectories
    elseif paths == true && data == false
        return trajectories
    end
end

const _JUMP_SOLVER_MAX_ITERS = 80

@inline function _materialize_covariates(row::DataFrameRow, hazards::AbstractVector{<:_Hazard})
    cache = Vector{NamedTuple}(undef, length(hazards))
    for (i, hazard) in enumerate(hazards)
        cache[i] = extract_covariates(row, hazard.parnames)
    end
    return cache
end

"""
    _find_jump_time_bisection(gap_fn, lo, hi, delta_t;
                              value_tol = 1e-10,
                              max_iters = _JUMP_SOLVER_MAX_ITERS)

Solve `gap_fn(t) = 0` on `[lo, hi]` via bisection. The caller guarantees
`gap_fn(lo) ≤ 0` and `gap_fn(hi) ≥ 0`. Returns the midpoint once either the
function value drops below `value_tol` or the bracket width shrinks beneath
`max(delta_t, value_tol)`.
"""
function _find_jump_time_bisection(gap_fn, lo, hi, delta_t;
                                   value_tol = 1e-10,
                                   max_iters = _JUMP_SOLVER_MAX_ITERS)
    lo >= hi && return hi
    initial_lo, initial_hi = lo, hi
    lo_val = gap_fn(lo)
    lo_val > 0 && return lo
    hi_val = gap_fn(hi)
    hi_val < 0 && error("simulate_path failed to bracket jump time on interval [$initial_lo, $initial_hi]")

    for iter in 1:max_iters
        mid = 0.5 * (lo + hi)
        mid_val = gap_fn(mid)

        if !isfinite(mid_val)
            error("simulate_path encountered non-finite objective while locating jump time on interval [$initial_lo, $initial_hi]")
        end

        if abs(mid_val) <= value_tol || (hi - lo) <= max(delta_t, value_tol)
            return mid
        elseif mid_val > 0
            hi = mid
        else
            lo = mid
        end
    end

    error("simulate_path failed to locate jump time after $(max_iters) bisection iterations on interval [$initial_lo, $initial_hi]")
end

@inline function _sample_next_state(rng::AbstractRNG, probs::AbstractVector{Float64}, trans_inds::AbstractVector{Int})
    threshold = rand(rng)
    cumulative = 0.0
    for idx in trans_inds
        cumulative += probs[idx]
        if threshold <= cumulative
            return idx
        end
    end
    return trans_inds[end]
end

"""
    simulate_path(model::MultistateProcess, subj::Int64, deltamin; optimize_fn = Optim.optimize)

Simulate a single sample path.

# Arguments 
- `model`: multistate model object
- `subj`: subject index
- `optimize_fn`: function with the same signature as `Optim.optimize` used to locate jump times (primarily for testing)
- `time_transform`: toggle Tang-style shared-trajectory solver. Defaults to `true` so the simulator matches exact-path likelihood mechanics but can be disabled for regression tests or debugging.
- `rng`: optional random-number generator. Defaults to the task-local RNG so callers can supply thread-local or reproducible generators.
"""
function simulate_path(model::MultistateProcess, subj::Int64, delta_u, delta_t;
                       optimize_fn = Optim.optimize,
                       time_transform::Bool = true,
                       rng::AbstractRNG = Random.default_rng())

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # current index
    row = 1 # row in subject's data that is incremented
    ind = subj_inds[row] # index in complete dataset
    subjdat_row = subj_dat[row, :] # current DataFrameRow for covariate extraction
    covars_cache = _materialize_covariates(subjdat_row, model.hazards)

    # current state
    scur = subj_dat.statefrom[1]

    tt_context = time_transform ? maybe_time_transform_context(model.parameters, subj_dat, model.hazards; time_column = :tstop) : nothing

    # tcur and tstop
    tcur    = subj_dat.tstart[1]
    tstop   = subj_dat.tstop[1]

    # initialize time in state and cumulative incidence 
    # clock resets after each transition
    timeinstate = 0.0
    cuminc = 0.0

    # vector for next state transition probabilities
    nstates  = size(model.tmat, 2)
    ns_probs = zeros(Float64, nstates)
    transitions_by_state = [findall(model.tmat[row, :] .!= 0) for row in 1:nstates]

    # initialize sample path
    times  = [tcur]; sizehint!(times, nstates * 2)
    states = [scur]; sizehint!(states, nstates * 2)

    # optimization settings
    atol = sqrt(eps())
    rtol = sqrt(atol)
    optmethod = Brent()

    # flag for whether to stop simulation
    # obviously don't simulate if the initial state is absorbing
    keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)
    
    # sample the cumulative incidence if transient
    if keep_going
        u = max(rand(rng), delta_u)
    end

    # simulate path
    while keep_going
        use_transform = time_transform && _time_transform_enabled(model.totalhazards[scur], model.hazards)
        
        # calculate event probability over the next interval
        interval_incid = (1 - cuminc) * (1 - survprob(
            timeinstate,
            timeinstate + tstop - tcur,
            model.parameters,
            covars_cache,
            model.totalhazards[scur],
            model.hazards;
            give_log = false,
            apply_transform = use_transform,
            cache_context = tt_context))

        # check if event happened in the interval
        if (u < (cuminc + interval_incid)) && (u >= cuminc)

            interval_len = tstop - tcur
            current_timeinstate = timeinstate
            gap_fn = t -> begin
                surv = survprob(
                    current_timeinstate,
                    current_timeinstate + t,
                    model.parameters,
                    covars_cache,
                    model.totalhazards[scur],
                    model.hazards;
                    give_log = false,
                    apply_transform = use_transform,
                    cache_context = tt_context)
                cuminc + (1 - cuminc) * (1 - surv) - u
            end

            if optimize_fn === Optim.optimize
                timeincrement = _find_jump_time_bisection(gap_fn, delta_t, interval_len, delta_t)
            else
                result = optimize_fn(t -> ((log(cuminc + (1 - cuminc) * (1 - survprob(
                    current_timeinstate,
                    current_timeinstate + t[1],
                    model.parameters,
                    covars_cache,
                    model.totalhazards[scur],
                    model.hazards;
                    give_log = false,
                    apply_transform = use_transform,
                    cache_context = tt_context))) - log(u))^2), delta_t, interval_len, optmethod; rel_tol = rtol, abs_tol = atol)

                if Optim.converged(result)
                    timeincrement = result.minimizer
                else
                    iters = Optim.iterations(result)
                    bracket = (delta_t, interval_len)
                    error("simulate_path failed to locate jump time after $iters iterations on interval $bracket")
                end
            end

            timeinstate += timeincrement

            # calculate next state transition probabilities 
            # next_state_probs!(ns_probs, timeinstate, scur, subjdat_row, model.parameters, model.hazards, model.totalhazards, model.tmat)

            trans_inds = transitions_by_state[scur]
            next_state_probs!(
                ns_probs,
                trans_inds,
                timeinstate,
                scur,
                covars_cache,
                model.parameters,
                model.hazards,
                model.totalhazards;
                apply_transform = time_transform,
                cache_context = tt_context)
            scur = _sample_next_state(rng, ns_probs, trans_inds)
            
            # increment time in state and cache the jump time and state
            tcur = times[end] + timeinstate
            push!(times, tcur)
            push!(states, scur)
                
            # check if the next state is transient
            keep_going = isa(model.totalhazards[scur], _TotalHazardTransient) 

            # draw new cumulative incidence, reset cuminc and time in state
            if keep_going
                u           = max(rand(rng), delta_u) # sample cumulative incidence
                cuminc      = 0.0 # reset cuminc
                timeinstate = 0.0 # reset time in state
            end

        elseif u >= (cuminc + interval_incid) # no transition in interval

            # if you keep going do some bookkeeping
            if row != size(subj_dat, 1)  # no censoring

                # increment the time in state
                timeinstate += tstop - tcur

                # increment cumulative inicidence
                cuminc += interval_incid

                # increment the row indices and interval endpoints
                row  += 1
                ind  += 1
                subjdat_row = subj_dat[row, :]  # Update DataFrameRow for new interval
                covars_cache = _materialize_covariates(subjdat_row, model.hazards)
                tcur  = subj_dat.tstart[row]
                tstop = subj_dat.tstop[row]

            else # censoring, return current state and tmax
                # stop sampling
                keep_going = false

                # increment tcur 
                tcur = tstop

                # push the state and time at the right endpoint
                push!(times, tcur)
                push!(states, scur)     
            end
        end
    end

    return SamplePath(subj, times, states)
end

