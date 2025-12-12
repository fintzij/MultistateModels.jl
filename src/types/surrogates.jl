# =============================================================================
# Surrogate Type Definitions
# =============================================================================
# Types for Markov surrogates and phase-type surrogates used in MCEM.
# =============================================================================

"""
    MarkovSurrogate(hazards::Vector{_MarkovHazard}, parameters::NamedTuple; fitted::Bool=false)

Markov surrogate for importance sampling proposals in MCEM.
Uses ParameterHandling.jl for parameter management.

# Fields
- `hazards::Vector{_MarkovHazard}`: Exponential hazard functions for each transition
- `parameters::NamedTuple`: Parameter structure (flat, nested, natural, unflatten)
- `fitted::Bool`: Whether the surrogate parameters have been fitted via MLE.
  If `false`, parameters are default/placeholder values and the surrogate should be
  fitted before use in MCEM or importance sampling.

# Construction
```julia
# Unfitted surrogate (needs fitting before use)
surrogate = MarkovSurrogate(hazards, params)  # fitted=false by default

# Fitted surrogate
surrogate = MarkovSurrogate(hazards, params; fitted=true)
```

See also: [`set_surrogate!`](@ref)
"""
struct MarkovSurrogate
    hazards::Vector{_MarkovHazard}
    parameters::NamedTuple
    fitted::Bool
    
    # Inner constructor that accepts any vector of hazards (converts to _MarkovHazard)
    function MarkovSurrogate(hazards::Vector{<:_Hazard}, parameters::NamedTuple; fitted::Bool=false)
        # Verify all hazards are Markov-compatible
        for h in hazards
            h isa _MarkovHazard || throw(ArgumentError(
                "MarkovSurrogate requires all hazards to be Markov (exponential). Got $(typeof(h))."))
        end
        new(convert(Vector{_MarkovHazard}, hazards), parameters, fitted)
    end
end

"""
    SurrogateControl(model::MultistateProcess, statefrom, targets, uinds, ginds)

Struct containing objects for computing the discrepancy of a Markov surrogate.
"""
struct SurrogateControl
    model::MultistateProcess
    statefrom::Int64
    targets::Matrix{Float64}
    uinds::Vector{Union{Nothing, Int64}}
    ginds::Vector{Union{Nothing, Int64}}
end
