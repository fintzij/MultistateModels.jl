"""
    observe_path(samplepath::SamplePath, model::MultistateModel, ind::Int64)

Return `statefrom` and `stateto` for a jump chain observed at `tstart` and `tstop`.
"""
# function observe_path(samplepath::SamplePath, model::MultistateModel, subj::Int64)

#     # grab the subject's data as a view
#     subj_inds = model.subjectindices[subj]
#     subj_dat = view(model.data, subj_inds, :)

#     # create a matrix for the state sequence
#     stateseq = similar(subj_dat[:,[:statefrom, :stateto]])

#     # initialize current time and time interval
#     tcur   = samplepath.times[1]
 
#     # loop through subj_dat
#     for r in Base.OneTo(nrow(subj_dat))

#         # get interval endpoints
#         tstart = subj_dat.tstart[r]
#         tstop  = subj_dat.tstop[r]

#         if(tcur )
#     end

# end


"""
    curate_path()


"""
# some function to curate a discretely observed sample path to add back other data.
# function curate_path(observedpath, model, subj)
# end