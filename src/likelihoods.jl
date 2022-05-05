function loglik_samplepath(path::SamplePath, model::MultistateModel)

    # initialize log likelihood
    loglik = 0.0

    # number of jumps
    n_intervals = length(path.times) - 1

    # subject data
    subj_inds = model.subjectindices[path.subj]
    subj_dat  = view(model.data, subj_inds, :)

    # current index
    row = 1 # row in subject data
    ind = subj_inds[row] # index in complete data

    # current and next need this?
    scur  = path.states[1]
    snext = path.states[2]

    # interval tcur and tstop
    tcur  = subj_dat.tstart[1]
    tstop = subj_dat.tstop[1]

    # recurse through the sample path
    for i in Base.OneTo(n_intervals)

    end

end