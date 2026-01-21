"""
    make_constraints(cons::Vector{Expr}, lcons::Vector{Float64}, ucons::Vector{Float64})

Create a constraints specification for model fitting.

# Throws
- `ArgumentError` if vectors have different lengths
"""
function make_constraints(;cons::Vector{Expr}, lcons::Vector{Float64}, ucons::Vector{Float64})
    if !(length(cons) == length(lcons) == length(ucons))
        throw(ArgumentError("cons ($(length(cons))), lcons ($(length(lcons))), and ucons ($(length(ucons))) must all be the same length."))
    end
    return (cons = cons, lcons = lcons, ucons = ucons)
end

"""
    parse_constraints(cons, hazards; consfun_name = :consfun)

Parse user-defined constraints to generate a function constraining parameters for use in the solvers provided by Optimization.jl.
"""
function parse_constraints(cons::Vector{Expr}, hazards; consfun_name = :consfun_model)

    # grab parameter names
    pars_flat = reduce(vcat, map(x -> x.parnames, hazards))

    # loop through the parameters 
    for h in eachindex(cons)
        # grab the expression
        ex = cons[h]

        # sub out parameter symbols for indices
        for p in eachindex(pars_flat)  
            ex = MacroTools.postwalk(ex -> ex == pars_flat[p] ? :(parameters[$p]) : ex, ex)
        end
        
        cons[h] = ex
    end

    # manually construct the body of the constraint function
    cons_body = Meta.parse("res .= " * string(:[$(cons...),]))
    cons_call = Expr(:call, consfun_name, :(res), :(parameters), :(data))
    consfun_name = Expr(:function, cons_call, cons_body)

    # evaluate to compile the function
    @RuntimeGeneratedFunction(consfun_name)
end

# =============================================================================
# Memory Management
# =============================================================================

"""
    clear_all_workspaces!()

Clear all thread-local workspaces to free memory.

This function clears:
- TVC interval workspaces (used for time-varying covariate evaluation)
- Path sampling workspaces (used for MCEM inference)

Call this function in long-running processes after model fitting is complete
to reclaim memory. All workspaces will be lazily re-created on next use.

# Example
```julia
# Fit multiple models in a loop
for i in 1:100
    model = multistatemodel(...)
    fitted = fit(model)
    process_results(fitted)
end

# Free accumulated workspace memory
clear_all_workspaces!()
```

See also: [`clear_tvc_workspaces!`](@ref), [`clear_path_workspaces!`](@ref)
"""
function clear_all_workspaces!()
    clear_tvc_workspaces!()
    clear_path_workspaces!()
    return nothing
end
