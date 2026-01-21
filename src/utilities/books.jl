# ============================================================================
# Bookkeeping and Data Container Functions
# ============================================================================
# Functions for building transition probability matrix (TPM) containers,
# hazard matrix books, TPM mappings, forward-backward matrices, data 
# collapsing, surrogate initialization, and subject data creation.
# ============================================================================

"""
    build_tpm_book(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

Build container for holding transition probability matrices.
"""
function build_tpm_book(T::DataType, tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

    # build the TPM container
    nstates = size(tmat, 1)
    nmats   = map(x -> nrow(x), tpm_index) 
    book    = [[zeros(T, nstates, nstates) for j in 1:nmats[i]] for i in eachindex(tpm_index)]

    return book
end

"""
    build_hazmat_book(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

Build container for holding transition intensity matrices.
"""
function build_hazmat_book(T::DataType, tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})
    # Making this "type aware" by using T::DataType so that autodiff worksA
    # build the TPM container
    nstates = size(tmat, 1)
    nmats   = map(x -> nrow(x), tpm_index) 
    book    = [zeros(T, nstates, nstates) for j in eachindex(tpm_index)]

    return book
end

"""
    build_tpm_mapping(data::DataFrame)

Construct bookkeeping objects for transition probability matrices for time intervals over which a multistate Markov process is piecewise homogeneous. The first bookkeeping object is a data frame that 
"""
function build_tpm_mapping(data::DataFrame) 

    # maps each row in dataset to TPM
    # first col is covar combn, second is tpm index
    tpm_map = zeros(Int, nrow(data), 2)

    # check if the data contains covariates
    if ncol(data) == 6 # no covariates
        
        # get intervals
        gaps = data.tstop - data.tstart

        # get unique start and stop
        ugaps = sort(unique(gaps))

        # for solving Kolmogorov equations - saveats
        tpm_index = 
            [DataFrame(tstart = 0,
                       tstop  = ugaps,
                       datind = 0),]

        # first instance of each interval in the data
        for i in Base.OneTo(nrow(tpm_index[1]))
            tpm_index[1].datind[i] = 
                findfirst(gaps .== tpm_index[1].tstop[i])
        end

        # match intervals to unique tpms
        tpm_map[:,1] .= 1
        for i in Base.OneTo(size(tpm_map, 1))
            tpm_map[i,2] = findfirst(ugaps .== gaps[i])
        end    

    else
        # get unique covariates
        covars = data[:,Not(1:6)]
        ucovars = unique(data[:,Not(1:6)])

        # get gap times
        gaps = data.tstop - data.tstart

        # initialize tpm_index
        tpm_index = [DataFrame() for i in 1:nrow(ucovars)]

        # for each set of unique covariates find gaps
        for k in Base.OneTo(nrow(ucovars))

            # get indices for rows that have the covars
            covinds = findall(map(x -> all(x == ucovars[k,:]), eachrow(covars)) .== 1)

            # find unique gaps 
            ugaps = sort(unique(gaps[covinds]))

            # fill in tpm_index
            tpm_index[k] = DataFrame(tstart = 0, tstop = ugaps, datind = 0)

            # first instance of each interval in the data
            for i in Base.OneTo(nrow(tpm_index[k]))
                tpm_index[k].datind[i] = 
                    covinds[findfirst(gaps[covinds] .== tpm_index[k].tstop[i])]
            end

            # fill out the tpm_map 
            # match intervals to unique tpms
            tpm_map[covinds, 1] .= k
            for i in eachindex(covinds)
                tpm_map[covinds[i],2] = findfirst(ugaps .== gaps[covinds[i]])
            end  
        end
    end

    # return objects
    return tpm_index, tpm_map
end

"""
    build_fbmats(model)

Build the forward recursion matrices.
"""
function build_fbmats(model)

    # get sizes of stuff
    n_states = size(model.tmat, 1)
    n_times = [sum(model.data.id .== s) for s in unique(model.data.id)]

    # create the forward matrices
    fbmats = [zeros(Float64, n_times[s], n_states, n_states) for s in eachindex(n_times)]

    return fbmats
end

"""
    collapse_data(data::DataFrame; SubjectWeights::Vector{Float64} = ones(unique(data.id)))

Collapse subjects to create an internal representation of a dataset and optionally recompute a vector of subject weights.
"""
function collapse_data(data::DataFrame; SubjectWeights::Vector{Float64} = ones(Float64, length(unique(data.id))))
    
    # find unique subjects
    ids = unique(data.id)
    _data = [DataFrame() for k in 1:length(ids)]
    for k in ids
        _data[k] = data[findall(data.id .== k),Not(:id)]
    end
    _DataCollapsed = unique(_data)

    # find the collapsed dataset for each individual
    inds = map(x -> findfirst(_DataCollapsed .== Ref(x)), _data)

    # tabulate the SubjectWeights
    SubjectWeightsCollapsed = map(x -> sum(SubjectWeights[findall(inds .== x)]), unique(inds))

    # add a fake id variable to the collapsed datasets (for purchasing alcohol)
    for k in 1:length(_DataCollapsed)
        insertcols!(_DataCollapsed[k], :tstart, :id => fill(k, nrow(_DataCollapsed[k])))
    end

    # vcat
    DataCollapsed = reduce(vcat, _DataCollapsed)
       
    return DataCollapsed, SubjectWeightsCollapsed
end

# NOTE: initialize_surrogate! is now defined in surrogate/markov.jl
# See that file for the unified entry point for surrogate creation.

"""
    make_subjdat(path::SamplePath, subjectdata::SubDataFrame)

Create a DataFrame for a single subject from a SamplePath object and the original data for that subject.
"""
function make_subjdat(path::SamplePath, subjectdata::SubDataFrame) 

    # times when the likelihood needs to be evaluated
    if (ncol(subjectdata) > 6) & (nrow(subjectdata) > 1)
        
        # times when covariates change
        keepvec = findall(map(i -> !isequal(subjectdata[i-1, 7:end], subjectdata[i, 7:end]), 2:nrow(subjectdata))) .+ 1 

        # utimes is the times in the path (which includes first and last times from subjectdata), and tstart when covariates change
        utimes = sort(unique(vcat(path.times, subjectdata.tstart[keepvec])))

    else 
        utimes = path.times
    end

    # get indices in the data object that correspond to the unique times
    datinds = searchsortedlast.(Ref(subjectdata.tstart), utimes)

    # get indices in the path that correspond to the unique times
    pathinds = searchsortedlast.(Ref(path.times), utimes)

    # make subject data
    subjdat_lik = DataFrame(
        tstart = utimes[Not(end)],
        tstop  = utimes[Not(begin)],
        increment = diff(utimes),
        sojourn = 0.0,
        sojournind = pathinds[Not(end)],
        statefrom = path.states[pathinds[Not(end)]],
        stateto   = path.states[pathinds[Not(begin)]])

    # compute sojourns
    subjdat_gdf = groupby(subjdat_lik, :sojournind)
    for g in subjdat_gdf
        g.sojourn .= cumsum([0.0; g.increment[Not(end)]])
    end

    # remove sojournind
    select!(subjdat_lik, Not(:sojournind))

    # tack on covariates if any
    if ncol(subjectdata) > 6
        # get covariates at the relevant times
        covars = subjectdata[datinds[Not(end)], Not([:id, :tstart, :tstop, :statefrom, :stateto, :obstype])]
        # concatenate and return
        subjdat_lik = hcat(subjdat_lik, covars)
    end
    
    # output
    return subjdat_lik
end
