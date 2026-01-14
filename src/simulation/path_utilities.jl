"""
    observe_path(samplepath::SamplePath, model::MultistateProcess;
                 obstype_by_transition=nothing, trans_map=nothing) 

Return `statefrom` and `stateto` for a jump chain observed at `tstart` and `tstop`.

# Arguments
- `samplepath::SamplePath`: Continuous-time sample path
- `model::MultistateProcess`: Model containing data with observation scheme
- `obstype_by_transition::Union{Nothing,Dict{Int,Int}}`: Optional mapping from
  transition index to observation type code.
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
    trans_map::Union{Nothing,Dict{Tuple{Int,Int},Int}} = nothing
)
    # Check if per-transition logic is needed
    use_per_transition = !isnothing(obstype_by_transition)
    
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
                                    trans_map, obstype_by_transition)

    # indices for panel data
    panel_inds = [1; map(x -> searchsortedlast(samplepath.times, x), subj_dat[:,:tstop])]

    # create a matrix for the state sequence
    obsdat = similar(subj_dat, nrows)
    
    # Initialize statefrom to missing - will be set explicitly for some rows
    # and propagated for others. Use missings() to ensure proper Union{Missing,Int} type.
    obsdat.statefrom = missings(Union{Missing, Int64}, nrows)
    
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
                obstype_by_transition
            )
        else
            # Panel/censored interval
            if use_per_transition
                # Per-transition logic: check for exact transitions within panel interval
                rowind = _process_panel_interval_with_per_transition!(
                    obsdat, rowind, r, samplepath, subj_dat,
                    trans_map, obstype_by_transition
                )
            else
                # Original behavior: preserve template obstype
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
    end

    # propagate tstop and state to to origin state and time
    # Note: tstart is always propagated, but statefrom is only propagated
    # for rows where it wasn't already explicitly set (marked as missing)
    obsdat.tstart[Not(1)] = obsdat.tstop[Not(end)]
    
    # Only propagate statefrom for rows where it's still missing
    # (Exact observations from panel intervals have statefrom set explicitly)
    for i in 2:nrow(obsdat)
        if ismissing(obsdat.statefrom[i])
            obsdat.statefrom[i] = obsdat.stateto[i-1]
        end
    end

    # set starting time and state
    obsdat.tstart[1] = samplepath.times[1]
    if ismissing(obsdat.statefrom[1])
        obsdat.statefrom[1] = samplepath.states[1]
    end

    # drop rows where subject starts in an absorbing state
    transient_states = findall(isa.(model.totalhazards, _TotalHazardTransient))
    keep_inds = map(x -> ((obsdat.statefrom[x] in transient_states) | ismissing(obsdat.statefrom[x])), collect(1:size(obsdat, 1)))

    # return state sequence
    return obsdat[keep_inds,:]
end

"""
    _count_observation_rows(samplepath, subj_dat, use_per_transition, trans_map,
                            obstype_by_transition)

Count the number of rows needed for observation data with per-transition obstypes.
"""
function _count_observation_rows(
    samplepath::SamplePath, 
    subj_dat,
    use_per_transition::Bool,
    trans_map::Union{Nothing,Dict{Tuple{Int,Int},Int}},
    obstype_by_transition::Union{Nothing,Dict{Int,Int}}
)
    nrows = 0
    for r in Base.OneTo(nrow(subj_dat))
        if subj_dat.obstype[r] != 1 && !use_per_transition
            # Non-exact interval without per-transition: always 1 row
            nrows += 1
        elseif subj_dat.obstype[r] != 1 && use_per_transition
            # Non-exact interval (panel/censored) with per-transition logic
            # May need additional rows for exact transitions within the interval
            nrows += _count_per_transition_rows(
                samplepath, subj_dat.tstart[r], subj_dat.tstop[r],
                trans_map, obstype_by_transition
            )
        elseif !use_per_transition
            # Exact interval without per-transition: original logic
            # Note: must guarantee at least 1 row even for zero-length intervals
            # (e.g., phase-type expansion reset rows with tstart == tstop)
            nrows += max(1, length(
                unique(
                    [subj_dat.tstart[r];
                     subj_dat.tstop[r];
                     samplepath.times[findall(subj_dat.tstart[r] .<= samplepath.times .<= subj_dat.tstop[r])]])) - 1)
        else
            # Exact interval with per-transition logic
            nrows += _count_per_transition_rows(
                samplepath, subj_dat.tstart[r], subj_dat.tstop[r],
                trans_map, obstype_by_transition
            )
        end            
    end
    return nrows
end

"""
    _count_per_transition_rows(samplepath, tstart, tstop, trans_map, obstype_by_transition)

Count rows for an interval with per-transition observation types.

Returns the number of rows to emit. The logic handles the important case where 
panel-observed transitions occur before exact-observed transitions within the 
same interval - in this case, a panel row must be emitted BEFORE the exact row(s)
to properly capture the subject's state history.

Row structure:
- If there are exact transitions AND there was time before the first exact:
  - 1 panel row from interval start to first exact transition  
  - 1 row for each exact transition
  - 1 final row to tstop (if there's time remaining)
- If only non-exact transitions: 1 row for the whole interval
"""
function _count_per_transition_rows(
    samplepath::SamplePath,
    tstart::Float64,
    tstop::Float64,
    trans_map::Dict{Tuple{Int,Int},Int},
    obstype_by_transition::Dict{Int,Int}
)
    # Find jumps in interval
    jump_inds = findall(tstart .< samplepath.times .< tstop)
    
    if isempty(jump_inds)
        # No transitions: single row for the interval
        return 1
    end
    
    # Classify transitions
    exact_jump_times = Float64[]
    has_non_exact = false
    
    for jump_idx in jump_inds
        statefrom = samplepath.states[jump_idx - 1]
        stateto = samplepath.states[jump_idx]
        
        # Skip if same state (no actual transition)
        statefrom == stateto && continue
        
        obtype = _get_transition_obstype(
            statefrom, stateto, trans_map, obstype_by_transition
        )
        
        if obtype == 1
            push!(exact_jump_times, samplepath.times[jump_idx])
        else
            has_non_exact = true
        end
    end
    
    n_exact = length(exact_jump_times)
    
    if n_exact == 0
        # Only non-exact transitions: single row for whole interval
        return 1
    end
    
    # Has exact transitions - count rows carefully
    # 
    # Key insight: exact rows need tstart < tstop (non-instantaneous) to work
    # with phase-type expansion. So we CANNOT emit a panel row ending at 
    # first_exact_time and then an exact row starting at first_exact_time.
    #
    # Instead, if there were non-exact (panel) transitions before the first exact,
    # we emit a panel row up to the LAST non-exact transition time (not first exact).
    # The exact row then spans from that point to the exact transition time.
    #
    # If there were NO non-exact transitions before first exact, the exact row
    # spans from tstart to the exact time.
    
    nrows = n_exact  # One row per exact transition
    
    # Check if there are non-exact transitions that occur BEFORE the first exact
    if has_non_exact
        # We'll emit a panel row for non-exact transitions
        nrows += 1
    end
    
    # If there's time after the last exact transition, need a final row
    last_exact_time = maximum(exact_jump_times)
    if last_exact_time < tstop
        nrows += 1  # Final row from last_exact_time to tstop
    end
    
    return nrows
end

"""
    _process_exact_interval!(obsdat, rowind, r, samplepath, subj_dat, 
                            use_per_transition, trans_map, obstype_by_transition)

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
    obstype_by_transition::Union{Nothing,Dict{Int,Int}}
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
            statefrom, stateto, trans_map, obstype_by_transition
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
    _process_panel_interval_with_per_transition!(obsdat, rowind, r, samplepath, subj_dat,
                                                  trans_map, obstype_by_transition)

Process a panel/censored observation interval when per-transition obstypes are specified.
Returns the updated row index.

This function handles the case where the template has obstype != 1 (panel or censored),
but obstype_by_transition specifies that some transitions should be recorded exactly.

The key insight is that when panel-observed transitions occur BEFORE exact-observed 
transitions within the same interval, we must emit a panel row BEFORE the exact row(s)
to properly capture the state the subject was in at the start of the interval.

Row emission logic:
1. If there's time before the first exact transition: emit panel row [tstart, first_exact_time]
2. For each exact transition: emit exact row ending at that transition time  
3. If there's time after the last exact transition: emit final row [last_exact_time, tstop]

# Arguments
- `obsdat`: Output DataFrame being filled
- `rowind`: Current row index in obsdat
- `r`: Current row index in subj_dat (template)
- `samplepath`: The simulated sample path
- `subj_dat`: Template data for this subject
- `trans_map`: Mapping from (statefrom, stateto) to transition index
- `obstype_by_transition`: Mapping from transition index to obstype code

# Returns
Updated row index after processing this interval.
"""
function _process_panel_interval_with_per_transition!(
    obsdat,
    rowind::Int,
    r::Int,
    samplepath::SamplePath,
    subj_dat,
    trans_map::Dict{Tuple{Int,Int},Int},
    obstype_by_transition::Dict{Int,Int}
)
    right_ind = searchsortedlast(samplepath.times, subj_dat.tstop[r])
    tstart = subj_dat.tstart[r]
    tstop = subj_dat.tstop[r]
    template_obstype = subj_dat.obstype[r]
    
    # Find jumps in this interval (strictly between tstart and tstop)
    jump_inds = findall(tstart .< samplepath.times .< tstop)
    
    if isempty(jump_inds)
        # No transitions in interval: single row with template obstype
        obsdat.tstop[rowind] = tstop
        
        if template_obstype == 2
            obsdat.stateto[rowind] = samplepath.states[right_ind]
        else
            obsdat.stateto[rowind] = missing
        end
        
        obsdat.obstype[rowind] = template_obstype
        
        if ncol(subj_dat) > 6
            obsdat[rowind, Not(1:6)] = subj_dat[r, Not(1:6)]
        end
        
        return rowind + 1
    end
    
    # Classify each jump as exact or non-exact based on obstype_by_transition
    # Track times for non-exact transitions as well
    exact_jumps = Tuple{Float64, Int}[]  # (time, jump_idx)
    non_exact_jumps = Tuple{Float64, Int, Int}[]  # (time, jump_idx, obstype)
    
    for jump_idx in jump_inds
        statefrom = samplepath.states[jump_idx - 1]
        stateto = samplepath.states[jump_idx]
        
        # Skip if same state (no actual transition)
        statefrom == stateto && continue
        
        obtype = _get_transition_obstype(
            statefrom, stateto, trans_map, obstype_by_transition
        )
        
        if obtype == 1
            push!(exact_jumps, (samplepath.times[jump_idx], jump_idx))
        else
            push!(non_exact_jumps, (samplepath.times[jump_idx], jump_idx, obtype))
        end
    end
    
    # Sort jumps by time
    sort!(exact_jumps, by = x -> x[1])
    sort!(non_exact_jumps, by = x -> x[1])
    
    current_row = rowind
    
    if isempty(exact_jumps)
        # No exact transitions: single row for whole interval with template obstype
        obsdat.tstop[current_row] = tstop
        
        if template_obstype == 2
            obsdat.stateto[current_row] = samplepath.states[right_ind]
        else
            obsdat.stateto[current_row] = missing
        end
        
        obsdat.obstype[current_row] = template_obstype
        
        if ncol(subj_dat) > 6
            obsdat[current_row, Not(1:6)] = subj_dat[r, Not(1:6)]
        end
        
        return current_row + 1
    end
    
    # Has exact transitions - need to emit rows carefully
    first_exact_time, first_exact_idx = exact_jumps[1]
    last_exact_time, _ = exact_jumps[end]
    
    # If there are non-exact transitions, emit a panel row first
    # This row covers all non-exact transitions and ensures proper statefrom propagation
    if !isempty(non_exact_jumps)
        # Find the last non-exact transition time
        last_non_exact_time, last_non_exact_idx, _ = non_exact_jumps[end]
        
        # Panel row ends at the last non-exact transition time
        obsdat.tstop[current_row] = last_non_exact_time
        obsdat.stateto[current_row] = samplepath.states[last_non_exact_idx]  # State after last non-exact
        obsdat.obstype[current_row] = maximum(x[3] for x in non_exact_jumps)  # Max obstype of non-exact
        # statefrom will be propagated from initial state
        
        if ncol(subj_dat) > 6
            obsdat[current_row, Not(1:6)] = subj_dat[r, Not(1:6)]
        end
        
        current_row += 1
    end
    
    # Emit exact rows for each exact transition
    # Each exact row spans from the previous row's tstop to the exact transition time
    # DO NOT explicitly set statefrom - let it propagate from previous row's stateto
    for (i, (jump_time, jump_idx)) in enumerate(exact_jumps)
        obsdat.tstop[current_row] = jump_time
        obsdat.stateto[current_row] = samplepath.states[jump_idx]
        # Note: statefrom is intentionally NOT set here
        # It will be propagated from the previous row's stateto (or initial state for first row)
        obsdat.obstype[current_row] = 1  # Exact observation
        
        if ncol(subj_dat) > 6
            obsdat[current_row, Not(1:6)] = subj_dat[r, Not(1:6)]
        end
        
        current_row += 1
    end
    
    # If there's time after the last exact transition, emit a final row
    if last_exact_time < tstop
        obsdat.tstop[current_row] = tstop
        
        # Determine obstype for final row:
        # - Use template obstype (typically panel)
        obsdat.obstype[current_row] = template_obstype
        
        if template_obstype == 2
            obsdat.stateto[current_row] = samplepath.states[right_ind]
        elseif template_obstype >= 3
            obsdat.stateto[current_row] = missing
        else
            # template_obstype == 1: exact - observe endpoint state
            obsdat.stateto[current_row] = samplepath.states[right_ind]
        end
        
        if ncol(subj_dat) > 6
            obsdat[current_row, Not(1:6)] = subj_dat[r, Not(1:6)]
        end
        
        current_row += 1
    end
    
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