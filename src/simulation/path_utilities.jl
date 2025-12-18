"""
    observe_path(samplepath::SamplePath, model::MultistateProcess;
                 obstype_by_transition=nothing, censoring_matrix=nothing, 
                 censoring_pattern=nothing, trans_map=nothing) 

Return `statefrom` and `stateto` for a jump chain observed at `tstart` and `tstop`.

# Arguments
- `samplepath::SamplePath`: Continuous-time sample path
- `model::MultistateProcess`: Model containing data with observation scheme
- `obstype_by_transition::Union{Nothing,Dict{Int,Int}}`: Optional mapping from
  transition index to observation type code. Takes precedence over censoring_matrix.
- `censoring_matrix::Union{Nothing,AbstractMatrix{Int}}`: Optional matrix of
  censoring codes (n_transitions × n_patterns)
- `censoring_pattern::Union{Nothing,Int}`: Column index in censoring_matrix to use
- `trans_map::Union{Nothing,Dict{Tuple{Int,Int},Int}}`: Pre-computed transition map
  (for efficiency when calling repeatedly)

# Observation Type Codes
- `1`: Exact - transition time and states fully observed
- `2`: Panel - only endpoint state at interval boundary observed
- `3+`: Censored with specific code - endpoint state missing

# Returns
DataFrame with columns: id, tstart, tstop, statefrom, stateto, obstype
"""
function observe_path(
    samplepath::SamplePath, 
    model::MultistateProcess;
    obstype_by_transition::Union{Nothing,Dict{Int,Int}} = nothing,
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}} = nothing,
    censoring_pattern::Union{Nothing,Int} = nothing,
    trans_map::Union{Nothing,Dict{Tuple{Int,Int},Int}} = nothing
)
    # Check if per-transition logic is needed
    use_per_transition = !isnothing(obstype_by_transition) || !isnothing(censoring_matrix)
    
    # Build transition map if needed and not provided
    if use_per_transition && isnothing(trans_map)
        trans_map = transition_index_map(model.tmat)
    end
    
    # get subjid
    subj = samplepath.subj

    # grab the subject's data as a view
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)

    # Pre-compute row counts for allocation
    # This is more complex with per-transition obstypes
    nrows = _count_observation_rows(samplepath, subj_dat, use_per_transition, 
                                    trans_map, obstype_by_transition, 
                                    censoring_matrix, censoring_pattern)

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
            # Original interval is exact - may apply per-transition logic
            rowind = _process_exact_interval!(
                obsdat, rowind, r, samplepath, subj_dat, 
                use_per_transition, trans_map, 
                obstype_by_transition, censoring_matrix, censoring_pattern
            )
        else
            # Panel/censored interval: preserve original behavior
            right_ind = searchsortedlast(samplepath.times, subj_dat.tstop[r])
            obsdat.tstop[rowind] = subj_dat.tstop[r]

            if subj_dat.obstype[r] == 2
                obsdat.stateto[rowind] = samplepath.states[right_ind]
            else
                obsdat.stateto[rowind] = missing
            end

            obsdat.obstype[rowind] = subj_dat.obstype[r]

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
    _count_observation_rows(samplepath, subj_dat, use_per_transition, trans_map,
                            obstype_by_transition, censoring_matrix, censoring_pattern)

Count the number of rows needed for observation data with per-transition obstypes.
"""
function _count_observation_rows(
    samplepath::SamplePath, 
    subj_dat,
    use_per_transition::Bool,
    trans_map::Union{Nothing,Dict{Tuple{Int,Int},Int}},
    obstype_by_transition::Union{Nothing,Dict{Int,Int}},
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}},
    censoring_pattern::Union{Nothing,Int}
)
    nrows = 0
    for r in Base.OneTo(nrow(subj_dat))
        if subj_dat.obstype[r] != 1
            # Non-exact interval: always 1 row
            nrows += 1
        elseif !use_per_transition
            # Exact interval without per-transition: original logic
            nrows += length(
                unique(
                    [subj_dat.tstart[r];
                     subj_dat.tstop[r];
                     samplepath.times[findall(subj_dat.tstart[r] .<= samplepath.times .<= subj_dat.tstop[r])]])) - 1
        else
            # Exact interval with per-transition logic
            nrows += _count_per_transition_rows(
                samplepath, subj_dat.tstart[r], subj_dat.tstop[r],
                trans_map, obstype_by_transition, censoring_matrix, censoring_pattern
            )
        end            
    end
    return nrows
end

"""
    _count_per_transition_rows(samplepath, tstart, tstop, trans_map,
                               obstype_by_transition, censoring_matrix, censoring_pattern)

Count rows for an exact interval with per-transition observation types.

Returns the number of rows to emit:
- One row for each exact transition (at exact transition time)
- One final row covering interval to tstop (with obstype from non-exact transitions if any)
"""
function _count_per_transition_rows(
    samplepath::SamplePath,
    tstart::Float64,
    tstop::Float64,
    trans_map::Dict{Tuple{Int,Int},Int},
    obstype_by_transition::Union{Nothing,Dict{Int,Int}},
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}},
    censoring_pattern::Union{Nothing,Int}
)
    # Find jumps in interval
    jump_inds = findall(tstart .< samplepath.times .< tstop)
    
    if isempty(jump_inds)
        # No transitions: single row for the interval
        return 1
    end
    
    # Count exact transitions
    n_exact = 0
    
    for jump_idx in jump_inds
        statefrom = samplepath.states[jump_idx - 1]
        stateto = samplepath.states[jump_idx]
        
        # Skip if same state (no actual transition)
        statefrom == stateto && continue
        
        obtype = _get_transition_obstype(
            statefrom, stateto, trans_map,
            obstype_by_transition, censoring_matrix, censoring_pattern
        )
        
        if obtype == 1
            n_exact += 1
        end
    end
    
    # Rows: 1 for each exact transition + 1 final row to tstop
    return n_exact + 1
end

"""
    _process_exact_interval!(obsdat, rowind, r, samplepath, subj_dat, 
                            use_per_transition, trans_map,
                            obstype_by_transition, censoring_matrix, censoring_pattern)

Process an exact observation interval, potentially with per-transition obstypes.
Returns the updated row index.

With per-transition obstypes:
- Each EXACT transition emits a row ending at the exact transition time
- One final row covers the remaining interval to tstop, with obstype = max(non-exact obstypes) 
  if any, or 1 (exact) if all transitions are exact or there are no transitions
- For non-exact final rows: panel (obstype=2) observes endpoint state, 
  censored (obstype>=3) has missing endpoint state
"""
function _process_exact_interval!(
    obsdat,
    rowind::Int,
    r::Int,
    samplepath::SamplePath,
    subj_dat,
    use_per_transition::Bool,
    trans_map::Union{Nothing,Dict{Tuple{Int,Int},Int}},
    obstype_by_transition::Union{Nothing,Dict{Int,Int}},
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}},
    censoring_pattern::Union{Nothing,Int}
)
    right_ind = searchsortedlast(samplepath.times, subj_dat.tstop[r])
    jump_inds = findall(subj_dat.tstart[r] .< samplepath.times .< subj_dat.tstop[r])
    
    if !use_per_transition
        # Original exact observation logic
        nrows_local = 1 + length(jump_inds)
        obsdat_inds = range(rowind; length = nrows_local)

        obsdat.tstop[obsdat_inds[end]] = subj_dat.tstop[r]
        if !isempty(jump_inds)
            obsdat.tstop[obsdat_inds[Not(end)]] = samplepath.times[jump_inds]
        end
        
        obsdat.stateto[obsdat_inds[end]] = samplepath.states[right_ind]
        if !isempty(jump_inds)
            obsdat.stateto[obsdat_inds[Not(end)]] = samplepath.states[jump_inds]
        end
                
        obsdat.obstype[obsdat_inds] .= subj_dat.obstype[r]

        if ncol(subj_dat) > 6
            obsdat[obsdat_inds, Not(1:6)] = 
                subj_dat[r*ones(Int32, length(obsdat_inds)), Not(1:6)]
        end

        return rowind + length(obsdat_inds)
    end
    
    # Per-transition logic
    # Classify each jump
    exact_jumps = Int[]
    non_exact_obstypes = Int[]
    
    for jump_idx in jump_inds
        statefrom = samplepath.states[jump_idx - 1]
        stateto = samplepath.states[jump_idx]
        
        # Skip if same state
        statefrom == stateto && continue
        
        obtype = _get_transition_obstype(
            statefrom, stateto, trans_map,
            obstype_by_transition, censoring_matrix, censoring_pattern
        )
        
        if obtype == 1
            push!(exact_jumps, jump_idx)
        else
            push!(non_exact_obstypes, obtype)
        end
    end
    
    # Calculate number of rows to emit
    # Each exact jump gets its own row, plus one final row to tstop
    n_exact_rows = length(exact_jumps)
    total_rows = n_exact_rows + 1  # exact jumps + final segment row
    
    current_row = rowind
    
    # Emit exact rows for each exact jump
    for jump_idx in exact_jumps
        obsdat.tstop[current_row] = samplepath.times[jump_idx]
        obsdat.stateto[current_row] = samplepath.states[jump_idx]
        obsdat.obstype[current_row] = 1
        
        if ncol(subj_dat) > 6
            obsdat[current_row, Not(1:6)] = subj_dat[r, Not(1:6)]
        end
        
        current_row += 1
    end
    
    # Emit final row covering from last exact jump (or interval start) to tstop
    obsdat.tstop[current_row] = subj_dat.tstop[r]
    
    if !isempty(non_exact_obstypes)
        # Final row inherits the maximum non-exact obstype
        interval_obstype = maximum(non_exact_obstypes)
        obsdat.obstype[current_row] = interval_obstype
        
        if interval_obstype == 2
            # Panel: observe endpoint state
            obsdat.stateto[current_row] = samplepath.states[right_ind]
        else
            # Censored: endpoint state unknown
            obsdat.stateto[current_row] = missing
        end
    else
        # All transitions were exact, or no transitions occurred
        # Final row is exact observation at interval boundary
        obsdat.stateto[current_row] = samplepath.states[right_ind]
        obsdat.obstype[current_row] = 1
    end
    
    if ncol(subj_dat) > 6
        obsdat[current_row, Not(1:6)] = subj_dat[r, Not(1:6)]
    end
    current_row += 1
    
    return current_row
end

"""
    observe_path(samplepath::SamplePath, times::Vector{Float64}, ind::Int64) 

Return a vector with the states at the specified times.
"""
function observe_path(samplepath::SamplePath, times::Vector{Float64}) 

    samplepath.states[searchsortedlast.(Ref(samplepath.times), times)]
end


"""
    extract_paths(model::MultistateProcess)

Extract sample paths from a multistate model's data field and return an array of SamplePath objects. 

# Arguments
- model: multistate model object
"""
function extract_paths(model::MultistateProcess)

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

        # grab the path
        samplepaths[i] = reduce_jumpchain(SamplePath(i, times, states))
    end

    return samplepaths
end


"""
    extract_paths(data::DataFrame)

Extract sample paths from a multistate model's data field and return an array of SamplePath objects. 

# Arguments
- data: DataFrame with data from multistate model object.
"""
function extract_paths(data::DataFrame)

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

        # grab the path
        samplepaths[i] = reduce_jumpchain(SamplePath(i, times, states))
    end

    return samplepaths
end

"""
    extract_sojourns(statefrom, stateto, samplepaths::Vector{SamplePath}; type = "events")
"""
function extract_sojourns(statefrom, stateto, samplepaths::Vector{SamplePath})

    # initialize times
    times = Vector{Float64}()
    sizehint!(times, length(samplepaths))

    # accumulate sojourns
    for s in eachindex(samplepaths)
        for i in Base.OneTo(length(samplepaths[s].states) - 1)
            if (samplepaths[s].states[i] == statefrom) && (samplepaths[s].states[i + 1] == stateto)
                push!(times, samplepaths[s].times[i+1] - samplepaths[s].times[i])
            end
        end
    end

    unique!(sort!(times))

    return times
end

"""
    reduce_jumpchain!(path::SamplePath)

Reduce the jump chain for a sample path so it doesn't include non-transitions, e.g., SamplePath(1,[0.0, 0.5, 1.0], [1, 1, 1]) -> SamplePath(1, [0.0, 1.0], [1, 1]). 
"""
function reduce_jumpchain(path::SamplePath)

    # no need to reduce if jump chain is constant
    pathlen = length(path.states)
    if pathlen .== 2
        newpath = path
    else
        # run length encoding
        rlp = rle(path.states)
        jumpinds = unique([1; cumsum(rlp[2])[Not(end)] .+ 1; pathlen])

        newpath = SamplePath(path.subj, path.times[jumpinds], path.states[jumpinds])
    end

    return newpath
end

"""
    path_to_dataframe(path::SamplePath)

Convert a single SamplePath to a DataFrame with the full path (exact transition times).

# Arguments
- `path::SamplePath`: A sample path with subject id, times, and states

# Returns
- `DataFrame`: Dataset with columns `id`, `tstart`, `tstop`, `statefrom`, `stateto`, `obstype`
  where `obstype = 1` indicates exact observation of transitions.

See also: [`paths_to_dataset`](@ref)
"""
function path_to_dataframe(path::SamplePath)
    n = length(path.times) - 1
    if n == 0
        # Edge case: path with single time point (no intervals)
        return DataFrame(
            id = Int[],
            tstart = Float64[],
            tstop = Float64[],
            statefrom = Int[],
            stateto = Int[],
            obstype = Int[]
        )
    end
    
    # Use @view to avoid allocating copies of the slices
    return DataFrame(
        id = fill(path.subj, n),
        tstart = @view(path.times[1:end-1]),
        tstop = @view(path.times[2:end]),
        statefrom = @view(path.states[1:end-1]),
        stateto = @view(path.states[2:end]),
        obstype = fill(1, n)  # obstype=1 for exact observations
    )
end

"""
    paths_to_dataset(samplepaths::Vector{SamplePath})

Convert a collection of sample paths to a DataFrame containing the full paths
(exact transition times), not observed at discrete times.

# Arguments
- `samplepaths::Vector{SamplePath}`: Vector of sample paths

# Returns
- `DataFrame`: Vertically concatenated dataset with full path data from all paths.
  Columns: `id`, `tstart`, `tstop`, `statefrom`, `stateto`, `obstype` (always 1 for exact).

# Example
```julia
# After drawing sample paths
paths = [draw_samplepath(i, model, ...) for i in 1:nsubj]
dataset = paths_to_dataset(paths)
```

See also: [`path_to_dataframe`](@ref), [`observe_path`](@ref), [`simulate`](@ref)
"""
function paths_to_dataset(samplepaths::AbstractVector{SamplePath})
    # Pre-compute total rows for pre-allocation
    total_rows = sum(max(0, length(p.times) - 1) for p in samplepaths)
    
    if total_rows == 0
        return DataFrame(
            id = Int[],
            tstart = Float64[],
            tstop = Float64[],
            statefrom = Int[],
            stateto = Int[],
            obstype = Int[]
        )
    end
    
    # Pre-allocate arrays
    ids = Vector{Int}(undef, total_rows)
    tstarts = Vector{Float64}(undef, total_rows)
    tstops = Vector{Float64}(undef, total_rows)
    statefroms = Vector{Int}(undef, total_rows)
    statetos = Vector{Int}(undef, total_rows)
    obstypes = Vector{Int}(undef, total_rows)
    
    # Fill arrays in one pass
    row = 1
    @inbounds for path in samplepaths
        n = length(path.times) - 1
        if n > 0
            for i in 1:n
                ids[row] = path.subj
                tstarts[row] = path.times[i]
                tstops[row] = path.times[i+1]
                statefroms[row] = path.states[i]
                statetos[row] = path.states[i+1]
                obstypes[row] = 1
                row += 1
            end
        end
    end
    
    return DataFrame(
        id = ids,
        tstart = tstarts,
        tstop = tstops,
        statefrom = statefroms,
        stateto = statetos,
        obstype = obstypes
    )
end

"""
    paths_to_dataset(samplepaths::Matrix{SamplePath})

Convert a matrix of sample paths (subjects × simulations) to a vector of DataFrames
containing the full paths (exact transition times).

# Arguments
- `samplepaths::Matrix{SamplePath}`: Matrix of sample paths (nsubj × nsim)

# Returns
- `Vector{DataFrame}`: Vector of datasets, one per simulation

See also: [`path_to_dataframe`](@ref), [`observe_path`](@ref), [`simulate`](@ref)
"""
function paths_to_dataset(samplepaths::Matrix{SamplePath})
    nsim = size(samplepaths, 2)
    datasets = Vector{DataFrame}(undef, nsim)
    
    # Process each simulation column using the optimized Vector method
    for i in 1:nsim
        datasets[i] = paths_to_dataset(@view(samplepaths[:, i]))
    end
    
    return datasets
end

"""
    paths_to_dataset(samplepaths::Vector{Vector{SamplePath}})

Convert a vector of vectors of sample paths (one vector per simulation) to a vector 
of DataFrames containing the full paths (exact transition times).

This method handles the output format from `simulate(...; paths=true)`.

# Arguments
- `samplepaths::Vector{Vector{SamplePath}}`: Vector of vectors of sample paths (nsim vectors, each with nsubj paths)

# Returns
- `Vector{DataFrame}`: Vector of datasets, one per simulation

# Example
```julia
data, paths = simulate(model; nsim=10, paths=true)
full_path_data = paths_to_dataset(paths)
```

See also: [`path_to_dataframe`](@ref), [`simulate`](@ref)
"""
function paths_to_dataset(samplepaths::Vector{Vector{SamplePath}})
    return [paths_to_dataset(sim_paths) for sim_paths in samplepaths]
end