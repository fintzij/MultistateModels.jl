# simulation functions
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
"""
function simulate(model::MultistateProcess; nsim = 1, data = true, paths = false)

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
            samplepath = simulate_path(model, j)

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

"""
simulate_path(model::MultistateProcess, subj::Int64)

Simulate a single sample path.

# Arguments 
- model: multistate model object
- subj: subject index
"""
function simulate_path(model::MultistateProcess, subj::Int64)

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # current index
    row = 1 # row in subject's data that is incremented
    ind = subj_inds[row] # index in complete dataset

    # current state
    scur = subj_dat.statefrom[1]

    # tcur and tstop
    tcur    = subj_dat.tstart[1]
    tstop   = subj_dat.tstop[1]

    # initialize time in state and cumulative incidence 
    # clock resets after each transition
    timeinstate = 0.0
    cuminc = 0.0

    # vector for next state transition probabilities
    nstates  = size(model.tmat, 2)
    # ns_probs = zeros(nstates)

    # initialize sample path
    times  = [tcur]; sizehint!(times, nstates * 2)
    states = [scur]; sizehint!(states, nstates * 2)

    if any(isa.(model.hazards, _SplineHazard))
        atol = maximum([maximum(map(x -> (isa(x, _SplineHazard) ? x.meshsize : 0), model.hazards))^-2, sqrt(eps())])
        rtol = sqrt(atol)
        optmethod = GoldenSection()
    else
        atol = sqrt(eps())
        rtol = sqrt(atol)
        optmethod = Brent()
    end

    # flag for whether to stop simulation
    # obviously don't simulate if the initial state is absorbing
    keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)
    
    # sample the cumulative incidence if transient
    if keep_going
        u = maximum([rand(1)[1], sqrt(sqrt(eps()))])
    end

    # simulate path
    while keep_going
        
        # calculate event probability over the next interval
        interval_incid = (1 - cuminc) * (1 - survprob(timeinstate, timeinstate + tstop - tcur, model.parameters, ind, model.totalhazards[scur], model.hazards; give_log = false))

        # check if event happened in the interval
        if (u < (cuminc + interval_incid)) && (u >= cuminc)

            # update the current time - roughly 1 millisecond must pass if time is in minutes
            timeincrement = optimize(t -> ((log(cuminc + (1 - cuminc) * (1 - survprob(timeinstate, timeinstate + t[1], model.parameters, ind, model.totalhazards[scur], model.hazards; give_log = false))) - log(u))^2), 0.00001, tstop - tcur, optmethod; rel_tol = rtol, abs_tol = atol)

            if Optim.converged(timeincrement)
                timeinstate += timeincrement.minimizer
            else
                error("Failed to converge in $(Optim.iterations(nexttime)) iterations")
            end            

            # calculate next state transition probabilities 
            # next_state_probs!(ns_probs, timeinstate, scur, ind, model.parameters, model.hazards, model.totalhazards, model.tmat)

            # sample the next state
            scur = rand(Categorical(next_state_probs(timeinstate, scur, ind, model.parameters, model.hazards, model.totalhazards, model.tmat)))
            
            # increment time in state and cache the jump time and state
            tcur = times[end] + timeinstate
            push!(times, tcur)
            push!(states, scur)
                
            # check if the next state is transient
            keep_going = isa(model.totalhazards[scur], _TotalHazardTransient) 

            # draw new cumulative incidence, reset cuminc and time in state
            if keep_going
                u           = maximum([rand(1)[1], sqrt(sqrt(eps()))]) # sample cumulative incidence
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

