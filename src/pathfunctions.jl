"""
    observe_path(samplepath::SamplePath, model::MultistateProcess) 

Return `statefrom` and `stateto` for a jump chain observed at `tstart` and `tstop`.
"""
function observe_path(samplepath::SamplePath, model::MultistateProcess) 

    # get subjid
    subj = samplepath.subj

    # grab the subject's data as a view
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)

    # identify the size of the subject's data
    nrows = 0
    for r in Base.OneTo(nrow(subj_dat))
        if subj_dat.obstype[r] != 1
            nrows += 1
        else
            nrows += 
                length(
                    unique(
                        [subj_dat.tstart[r];
                         subj_dat.tstop[r];
                         samplepath.times[findall(subj_dat.tstart[r] .<= samplepath.times .<= subj_dat.tstop[r])]])) - 1
        end            
    end

    # indices for panel data
    panel_inds = [1; map(x -> searchsortedlast(samplepath.times, x), subj_dat[:,:tstop])]

    # create a matrix for the state sequence
    obsdat = similar(subj_dat, nrows)
    
    # fill out id
    obsdat.id .= subj_dat.id[1]

    # loop through subj_dat
    rowind = 1 # starting row index in obsdat    
    for r in Base.OneTo(nrow(subj_dat))

        if subj_dat.obstype[r] == 1 

            # indices in the sample path
            # left_ind = searchsortedlast(samplepath.times, subj_dat.tstart[r])
            right_ind = searchsortedlast(samplepath.times, subj_dat.tstop[r])
            jump_inds = 
                findall(subj_dat.tstart[r] .< samplepath.times .< subj_dat.tstop[r])

            # number of rows to populate
            nrows = 1 + length(jump_inds)

            # indices of rows to populate
            obsdat_inds = range(rowind; length = nrows)

            # get the times 
            obsdat.tstop[obsdat_inds[end]] = subj_dat.tstop[r]
            obsdat.tstop[obsdat_inds[Not(end)]] = samplepath.times[jump_inds]
            
            # get the state
            obsdat.stateto[obsdat_inds[end]] = samplepath.states[right_ind]
            obsdat.stateto[obsdat_inds[Not(end)]] = samplepath.states[jump_inds]
                    
            # populate the obstype column
            obsdat.obstype[obsdat_inds] .= subj_dat.obstype[r]

            # copy the covariates
            if ncol(subj_dat) > 6
                obsdat[obsdat_inds, Not(1:6)] = 
                    subj_dat[r*ones(Int32, length(obsdat_inds)), Not(1:6)]
            end

            # increment the row index
            rowind += length(obsdat_inds)
        else
            
            # get indices in the sample path
            # left_ind = searchsortedlast(samplepath.times, subj_dat.tstart[r])
            right_ind = searchsortedlast(samplepath.times, subj_dat.tstop[r])

            # get the times 
            # obsdat.tstart[rowind] = subj_dat.tstart[r]
            obsdat.tstop[rowind] = subj_dat.tstop[r]

            # get the states
            if subj_dat.obstype[r] == 2
                obsdat.stateto[rowind] = samplepath.states[right_ind]

            else
                obsdat.stateto[rowind] = missing
            end

            # populate the obstype column
            obsdat.obstype[rowind] = subj_dat.obstype[r]

            # copy the covariates
            if ncol(subj_dat) > 6
                obsdat[rowind, Not(1:6)] = subj_dat[r, Not(1:6)]
            end

            rowind += 1
        end
    end

    # propagate tstop and state to to origin state and time
    obsdat.tstart[Not(1)] = obsdat.tstop[Not(end)]
    obsdat.statefrom[Not(1)] = obsdat.stateto[Not(end)]

    # set starting time and state
    obsdat.tstart[1] = samplepath.times[1]
    obsdat.statefrom[1] = samplepath.states[1]

    # drop rows where subject starts in an absorbing state
    transient_states = findall(isa.(model.totalhazards, _TotalHazardTransient))
    keep_inds = map(x -> ((obsdat.statefrom[x] in transient_states) | ismissing(obsdat.statefrom[x])), collect(1:size(obsdat, 1)))

    # return state sequence
    return obsdat[keep_inds,:]
end

"""
    observe_path(samplepath::SamplePath, times::Vector{Float64}, ind::Int64) 

Return a vector with the states at the specified times.
"""
function observe_path(samplepath::SamplePath, times::Vector{Float64}) 

    samplepath.states[searchsortedlast.(Ref(samplepath.times), times)]
end


"""
    extract_paths(model::MultistateProcess; self_transitions = false)

Extract sample paths from a multistate model's data field and return an array of SamplePath objects. 

# Arguments
- model: multistate model object
- self_transitions: keep self-transitions? Defaults to false.
"""
function extract_paths(model::MultistateProcess; self_transitions = false)

    # get IDs
    nsubj = length(model.subjectindices)

    # initialize array of sample paths
    samplepaths = Vector{SamplePath}(undef, nsubj)

    # grab the sample paths
    for i in eachindex(model.subjectindices)
        
        # get sequence of times
        times = [model.data[model.subjectindices[i], :tstart]; model.data[model.subjectindices[i][end], :tstop]]

        # get sequence of states
        states = [model.data[model.subjectindices[i], :statefrom]; model.data[model.subjectindices[i][end], :stateto]]

        # remove duplicates
        if !self_transitions && (length(states) > 2)
            if any(states[2:(end - 1)] .== states[1:(end-2)])
                dropinds = findall(states[2:(end-1)] .== states[1:(end-2)]) .+ 1

                states = states[Not(dropinds)]
                times = times[Not(dropinds)]
            end
        end

        # grab the path
        samplepaths[i] = SamplePath(i, times, states)
    end

    return samplepaths
end


"""
    extract_paths(model::MultistateProcess; self_transitions = false)

Extract sample paths from a multistate model's data field and return an array of SamplePath objects. 

# Arguments
- data: DataFrame with data from multistate model object.
- self_transitions: keep self-transitions? Defaults to false.
"""
function extract_paths(data::DataFrame; self_transitions = false)

    # get subject indices
    subjinds, nsubj = get_subjinds(data)
    
    # initialize array of sample paths
    samplepaths = Vector{SamplePath}(undef, nsubj)

    # grab the sample paths
    for i in Base.OneTo(nsubj)
        
        # get sequence of times
        times = [data[subjinds[i], :tstart]; data[subjinds[i][end], :tstop]]

        # get sequence of states
        states = [data[subjinds[i], :statefrom]; data[subjinds[i][end], :stateto]]

        # remove duplicates
        if !self_transitions && (length(states) > 2)
            if any(states[2:(end - 1)] .== states[1:(end-2)])
                dropinds = findall(states[2:(end-1)] .== states[1:(end-2)]) .+ 1

                states = states[Not(dropinds)]
                times = times[Not(dropinds)]
            end
        end

        # grab the path
        samplepaths[i] = SamplePath(i, times, states)
    end

    return samplepaths
end

"""
    extract_sojourns(hazard, data::DataFrame, samplepaths::Vector{SamplePath})

Extract unique gap and sojourn times.
"""
function extract_sojourns(hazard, data::DataFrame, samplepaths::Vector{SamplePath}; sojourns_only = true)

    # initialize times
    times = Vector{Float64}()
    sizehint!(times, length(samplepaths))

    for s in eachindex(samplepaths)
        
        # get subject data
        subj_dat = view(data, findall(data.id .== samplepaths[s].subj), :)

        for i in Base.OneTo(length(samplepaths[s].states)-1)
            if samplepaths[s].states[i] == hazard.statefrom

                if sojourns_only
                    append!(times, diff(samplepaths[s].times[i:(i+1)]))
                else
                    # append the times at which the hazard and cumulative hazard are evaluated
                    append!(times, unique([diff([samplepaths[s].times[i]; subj_dat.tstop[findall((subj_dat.tstop .> samplepaths[s].times[i]) .&& (subj_dat.tstop .< samplepaths[s].times[i+1]))]; samplepaths[s].times[i+1]]); diff(samplepaths[s].times[i:(i+1)])]))
                end
            end
        end
    end

    unique!(sort!(times))

    return times
end