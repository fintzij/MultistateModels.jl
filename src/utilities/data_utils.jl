# =============================================================================
# Data Utility Functions
# =============================================================================
#
# Utility functions for data preprocessing and manipulation.
#
# =============================================================================

"""
    center_covariates(data::DataFrame; exclude::Vector{Symbol}=Symbol[])

Center numeric covariates in a DataFrame by subtracting their means.

Standard multistate model columns (`:id`, `:tstart`, `:tstop`, `:statefrom`, 
`:stateto`, `:obstype`) are never centered. Additional columns can be excluded 
using the `exclude` argument.

# Arguments
- `data::DataFrame`: Input data (will be copied, not mutated)
- `exclude::Vector{Symbol}=Symbol[]`: Additional columns to exclude from centering

# Returns
A tuple `(centered_data, means)` where:
- `centered_data::DataFrame`: Copy of data with numeric covariates centered
- `means::NamedTuple`: Original means of centered columns (for recovering original values)

# Examples
```julia
# Basic usage
centered, means = center_covariates(data)

# Exclude additional columns
centered, means = center_covariates(data; exclude=[:binary_var, :id2])

# Recover original values
original_age = centered.age .+ means.age

# Use with model construction for interpretable baseline
centered, means = center_covariates(data)
model = multistatemodel(h12; data=centered)
# Now covariate coefficients are relative to mean, not zero
```

# Notes
- Only numeric columns (`eltype <: Real`) are centered
- Binary columns (0/1) are centered along with other numeric columns
- The function is useful for:
  - Making baseline hazard interpretable (corresponds to mean covariate values)
  - Improving numerical stability in optimization
  - Comparing models with different covariate scalings

See also: [`cumulative_incidence_at_reference`](@ref)
"""
function center_covariates(data::DataFrame; exclude::Vector{Symbol}=Symbol[])
    # Standard columns that are NEVER centered
    standard_cols = [:id, :tstart, :tstop, :statefrom, :stateto, :obstype]
    
    # Combine standard exclusions with user exclusions
    no_center = union(standard_cols, exclude)
    
    # Make a copy to avoid mutating input
    centered = copy(data)
    
    # Track means of centered columns
    means_dict = Dict{Symbol, Float64}()
    
    # Iterate through columns (convert String names to Symbol)
    for col_str in names(centered)
        col = Symbol(col_str)
        
        # Skip if in exclusion list
        col in no_center && continue
        
        # Skip non-numeric columns
        col_type = eltype(centered[!, col])
        (col_type <: Real || col_type <: Union{Missing, <:Real}) || continue
        
        # Handle missing values: compute mean of non-missing
        col_data = centered[!, col]
        if any(ismissing, col_data)
            non_missing = skipmissing(col_data)
            if isempty(non_missing)
                continue  # All missing, can't center
            end
            μ = mean(non_missing)
            centered[!, col] = [ismissing(x) ? missing : x - μ for x in col_data]
        else
            μ = mean(col_data)
            centered[!, col] = col_data .- μ
        end
        
        means_dict[col] = μ
    end
    
    # Convert means to NamedTuple for convenient access
    if isempty(means_dict)
        means = NamedTuple()
    else
        sorted_keys = sort(collect(keys(means_dict)))
        means = NamedTuple{Tuple(sorted_keys)}(Tuple(means_dict[k] for k in sorted_keys))
    end
    
    return centered, means
end

"""
    center_covariates(model::MultistateProcess; centering::Symbol=:mean)

Compute centering values for covariates used in a multistate model.

This function extracts all covariates used across hazards in the model and 
computes centering values (mean or median) based on the covariate values 
at the first observation for each unique subject.

# Arguments
- `model::MultistateProcess`: A fitted or unfitted multistate model
- `centering::Symbol=:mean`: Centering method, one of:
  - `:mean` - Center at sample mean (default)
  - `:median` - Center at sample median  

# Returns
A `NamedTuple` with covariate names as keys and centering values as Float64 values.
Returns an empty NamedTuple if the model has no covariates.

# Examples
```julia
model = multistatemodel(h12, h13; data=dat)

# Get mean covariate values for cumulative incidence computation
covar_means = center_covariates(model; centering=:mean)
ci = cumulative_incidence(t, model, covar_means; statefrom=1)

# Get median values (robust to outliers)
covar_medians = center_covariates(model; centering=:median)
```

# Notes
- Uses the first observation row for each subject to avoid counting repeated 
  covariate values multiple times
- The returned NamedTuple can be passed directly to `cumulative_incidence`

See also: [`cumulative_incidence`](@ref), [`cumulative_incidence_at_reference`](@ref)
"""
function center_covariates(model::MultistateProcess; centering::Symbol=:mean)
    # Validate centering argument
    centering in (:mean, :median) || 
        throw(ArgumentError("centering must be :mean or :median, got :$centering"))
    
    # Collect all covariate names from all hazards
    all_covar_names = Symbol[]
    for haz in model.hazards
        append!(all_covar_names, haz.covar_names)
    end
    unique!(all_covar_names)
    
    # Return empty NamedTuple if no covariates
    if isempty(all_covar_names)
        return NamedTuple()
    end
    
    # Extract covariate values from first row of each subject
    data = model.data
    subject_indices = model.subjectindices
    n_subjects = length(subject_indices)
    
    # Collect covariate values for each subject (use first row)
    covar_values = Dict{Symbol, Vector{Float64}}()
    for name in all_covar_names
        covar_values[name] = Float64[]
    end
    
    for subj in 1:n_subjects
        first_row_idx = first(subject_indices[subj])
        for name in all_covar_names
            push!(covar_values[name], Float64(data[first_row_idx, name]))
        end
    end
    
    # Compute centering values based on method
    centered_dict = Dict{Symbol, Float64}()
    for name in all_covar_names
        vals = covar_values[name]
        if centering == :mean
            centered_dict[name] = mean(vals)
        elseif centering == :median
            centered_dict[name] = median(vals)
        end
    end
    
    # Return as NamedTuple
    return NamedTuple{Tuple(all_covar_names)}(Tuple(centered_dict[name] for name in all_covar_names))
end


# =============================================================================
# Data Validation Functions
# =============================================================================

"""
    validate_data_continuity(df::DataFrame)

Check that multistate data has proper state continuity: statefrom[i+1] == stateto[i]
within each subject. Throws ArgumentError if validation fails.

Used primarily for test data validation to catch manual data construction errors
that can cause cryptic failures in MCEM/FFBS algorithms.

# Arguments
- `df::DataFrame`: Data with columns :id, :tstart, :tstop, :statefrom, :stateto

# Returns
- `true` if validation passes

# Throws
- `ArgumentError` with detailed message if state discontinuity is detected

# Examples
```julia
# Valid data
df_valid = DataFrame(
    id = [1, 1, 1],
    tstart = [0.0, 1.0, 2.0],
    tstop = [1.0, 2.0, 3.0],
    statefrom = [1, 1, 2],
    stateto = [1, 2, 3],
    obstype = [2, 2, 1]
)
validate_data_continuity(df_valid)  # Returns true

# Invalid data (discontinuity: stateto=1 but next statefrom=2)
df_invalid = DataFrame(
    id = [1, 1],
    tstart = [0.0, 1.0],
    tstop = [1.0, 2.0],
    statefrom = [1, 2],  # BUG: should be [1, 1]
    stateto = [1, 3],
    obstype = [2, 1]
)
validate_data_continuity(df_invalid)  # Throws ArgumentError
```

See also: [`simulate`](@ref), [`multistatemodel`](@ref)
"""
function validate_data_continuity(df::DataFrame)
    required_cols = [:id, :tstart, :statefrom, :stateto]
    for col in required_cols
        if !hasproperty(df, col)
            throw(ArgumentError("DataFrame missing required column: $col"))
        end
    end
    
    for subj_df in groupby(df, :id)
        # Sort by tstart within subject
        rows = sort(subj_df, :tstart)
        subj_id = rows.id[1]
        
        for i in 2:nrow(rows)
            prev_stateto = rows.stateto[i-1]
            curr_statefrom = rows.statefrom[i]
            
            # Skip validation for censored observations where stateto = 0
            if prev_stateto == 0
                continue
            end
            
            if curr_statefrom != prev_stateto
                throw(ArgumentError(
                    "State discontinuity in subject $subj_id at row $i: " *
                    "stateto[$(i-1)] = $prev_stateto, statefrom[$i] = $curr_statefrom. " *
                    "Each observation interval must start in the state where the previous interval ended."
                ))
            end
        end
    end
    
    return true
end
