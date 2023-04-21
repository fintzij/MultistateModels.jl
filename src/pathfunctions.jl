"""
    observe_path(samplepath::SamplePath, model::MultistateModel, ind::Int64)

Return `statefrom` and `stateto` for a jump chain observed at `tstart` and `tstop`.
"""
function observe_path(samplepath::SamplePath, model::MultistateModel, subj::Int64)

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
            if subj_dat.obstype[r] == 3
                obsdat.stateto[rowind] =
                    maximum(samplepath.states[panel_inds[r]:panel_inds[r+1]])

            elseif subj_dat.obstype[r] == 2
                obsdat.stateto[rowind] = samplepath.states[right_ind]

            elseif subj_dat.obstype[r] == 0
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
    keep_inds = map(x -> obsdat.statefrom[x] in transient_states, collect(1:size(obsdat, 1)))

    # return state sequence
    return obsdat[keep_inds,:]
end


"""
    extract_paths(model::MultistateModel)

Extract sample paths from a multistate model's data field and return an array of SamplePath objcets.
"""
function extract_paths(model::MultistateModel)

    # get IDs
    nsubj = length(model.subjectindices)

    # initialize array of sample paths
    samplepaths = Vector{SamplePath}(undef, nsubj)

    # grab the sample paths
    for i in eachindex(model.subjectindices)
        
        # grab the path
        samplepaths[i] = 
            SamplePath(
                i,
                [model.data[model.subjectindices[i], :tstart]; model.data[model.subjectindices[i][end], :tstop]],
                [model.data[model.subjectindices[i], :statefrom]; model.data[model.subjectindices[i][end], :stateto]])
    end

    return samplepaths
end
