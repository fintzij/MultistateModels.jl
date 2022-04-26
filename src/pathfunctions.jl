"""
    observe_path(samplepath::SamplePath, model::MultistateModel, ind::Int64)

Return `statefrom` and `stateto` for a jump chain observed at `tstart` and `tstop`.
"""
function observe_path(samplepath::SamplePath, model::MultistateModel, subj::Int64)

    # grab the subject's data as a view
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)

    # create a matrix for the state sequence
    stateseq = similar(subj_dat[:,[:statefrom, :stateto]])

    # initialize current time and time interval
    tcur = samplepath.times[1]

    # get mapping from sample path to subj_dat
    # last value of samplepath.times less than or equal to tstart
    inds_in_path = 
        map(x -> searchsortedlast(samplepath.times, x),
            subj_dat[:,:tstart])

    # loop through subj_dat
    for r in Base.OneTo(nrow(subj_dat))

        # missing data
        if subj_dat[r,:obstype] == 3

            stateseq[r, :stateto] = 
                maximum(samplepath.states[range(inds_in_path[max(1,r-1):r])])
            
        # 1 (exactly observed) or 2 (panel data) are the same    
        else 

           # state at the observation time
           stateseq[r, :stateto] = samplepath.states[inds_in_path[r]]

        end
    end

    # censor missing data
    stateseq[findall(subj_dat.obstype .== 0), :stateto] .= missing

    # set statefrom sequence
    stateseq[1,:statefrom] = samplepath.states[1]
    stateseq[Not(1), :statefrom] = stateseq[Not(end), :stateto]

    # return state sequence
    return stateseq
end


"""
    curate_path()


"""
# some function to curate a discretely observed sample path to add back other data.
# function curate_path(observedpath, model, subj)
# end