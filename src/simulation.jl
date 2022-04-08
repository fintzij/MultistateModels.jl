# simulation functions
# simone_path: simulates a single sample path
# sim_paths: wrapper around simeone_path to simulate nsamples x nids number of paths
# simulate: wrapper around sim_paths and maybe other stuff to simulate new data, also incorporates censoring, etc.

"""
    simulate(model::MultistateModel; n = 1, data = true, paths = false)

Simulate `n` datasets or collections of sample paths from a multistate model. If `data = true` (the default) discretely observed sample paths are returned, possibly subject to measurement error. If `paths = false` (the default), continuous-time sample paths are not returned.

# Arguments
- `model::MultistateModel`: object created by multistatemodel()
- `nsim`: number of sample paths to simulate
- `data`: boolean; if true then return discretely observed sample paths
- `paths`: boolean; if false then continuous-time sample paths not returned
"""
function simulate(model::MultistateModel; nsim = 1, data = true, paths = false)

    # throw an error if neither paths nor data are asked for
    if(paths == false & data == false)
        error("Why are you calling `simulate` if you don't want sample paths or data? Stop wasting my time.")
    end

    # number of subjects
    nsubj = length(model.subjectindices)

    # initialize array for simulated paths 
    if paths == true
        samplepaths = Array{SamplePath}(undef, nsubj, nsim)
    end

    # initialize container for simulated data
    # if data == true

    # end 

    for i in Base.OneTo(nsim)
        for j in Base.OneTo(nsubj)
            
            # simulate a path for subject j
            samplepath = simulate_path(model, j)

            # save path if requested
            if path == true
                samplepaths[j, i] = samplepath
            end

            # simulate data
            if data == true
                # simulate data
                # sampledata = simulate_data(samplepath)

                # save data
            end
        end
    end

    # return paths and data
    if paths == false & data == true
    elseif paths == true & data == true
    elseif paths == true & data == false
    end
end

"""
    simulate_path()

Simulate a single sample path.

# Arguments 
- model: multistate model object
- subj: subject index
"""
function simulate_path(model::MultistateModel, subj::Int64)

    # subset data for subject and exctract important bits
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)

    # current index
    row = 1 # row in subject's data that is incremented
    ind = subj_inds[row] # index in complete dataset

    # current state
    scur = subj_dat.statefrom[1]

    # tcur and tstop
    tcur    = subj_dat.tstart[1]
    tstop   = subj_dat.tstop[1]

    # initialize cumulative incidence - clock resets each transition
    cuminc = 0.0

    # vector for next state transition probabilities
    ns_probs = zeros(size(model.tmat,2))

    # initialize sample path
    times  = [tcur]
    states = [scur]

    # flag for whether to stop simulation
    # obviously don't simulate if the initial state is absorbing
    keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)
    
    # sample the cumulative incidence if transient
    if keep_going
        u = rand(1)[1]
    end

    # simulate path
    while keep_going
        
        # calculate event probability over the next interval
        interval_incid = 
            1 - survprob(model.totalhazards[scur], model.hazards, tcur, tstop, ind)

        # check if event happened in the interval
        if u < cuminc + interval_incid && u >= cuminc 

            # update the current time
            nexttime = optimize(
                t -> (logit(cuminc + (1 - MultistateModels.survprob(model.totalhazards[scur], model.hazards, tcur, t, ind))) - logit(u))^2, 
                tcur, tstop)

            if Optim.converged(nexttime)
                tcur = nexttime.minimizer
            else
                error("Failed to converge in $(Optim.iterations(nexttime)) iterations")
            end            

            # calculate next state transition probabilities 
            next_state_probs!(ns_probs, tcur, scur, ind, model)

            # sample the next state
            scur = rand(Categorical(ns_probs))
            
            # cache the jump time and state
            push!(times, tcur)
            push!(states, scur)
                
            # check if the next state is transient
            keep_going = 
                isa(model.totalhazards[scur], _TotalHazardTransient) 

            # draw new cumulative incidence and reset cuming
            if keep_going
                u = rand(1)[1] # sample cumulative incidence
                cuminc = 0.0 # reset cuminc
            end

        elseif u >= cuminc + interval_incid # no transition in interval
                            
            # if you keep going do some bookkeeping
            if row != size(subj_dat, 1)  # no censoring

                # increment the row indices and interval endpoints
                row  += 1
                ind  += 1
                tcur  = subj_dat.tstart[row]
                tstop = subj_dat.tstop[row]

                # increment cumulative inicidence
                cuminc += interval_incid

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

    return SamplePath(times, states)
end


"""
    next_state_probs!(ns_probs, scur, ind, model)

Update ns_probs with vector probabilities of transitioning to each state based on hazards from current state. ns_probs will then get fed into a multinomial sampler.

# Arguments 
- ns_probs: Vector of probabilities corresponding to each state, modified in place
- t: time at which hazards should be calculated
- scur: current state
- ind: index at complete dataset
- model: MultistateModel object
"""
function next_state_probs!(ns_probs, t, scur, ind, model)

    # set ns_probs to zero for impossible transitions
    ns_probs[findall(model.tmat[scur,:] .== 0.0)] .= 0.0

    # indices for possible destination states
    trans_inds = findall(model.tmat[scur,:] .!= 0.0)
        
    # calculate log hazards for possible transitions
    ns_probs[trans_inds] = 
        map(x -> call_haz(t, ind, x), model.hazards[model.totalhazards[scur].components])

    # normalize ns_probs
    ns_probs[trans_inds] = 
        softmax(ns_probs[model.totalhazards[scur].components])
end

function next_state_probs(t, scur, ind, model)

    # initialize vector of next state transition probabilities
    ns_probs = zeros(size(model.tmat, 2))

    # indices for possible destination states
    trans_inds = findall(model.tmat[scur,:] .!= 0.0)
    
    # calculate log hazards for possible transitions
    ns_probs[trans_inds] = 
        map(x -> call_haz(t, ind, x), model.hazards[model.totalhazards[scur].components])

    # return the next state transition probabilities
    ns_probs[trans_inds] = 
        softmax(ns_probs[model.totalhazards[scur].components])

    # return the next state probabilities
    return ns_probs
end

