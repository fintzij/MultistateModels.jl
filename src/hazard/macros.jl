const _ANALYTIC_HAZARD_FAMILY_ALIASES = Dict(
    :exp => :exp,
    :exponential => :exp,
    :wei => :wei,
    :weibull => :wei,
    :gom => :gom,
    :gompertz => :gom,
    :sp => :sp,
    :spline => :sp,
    :pt => :pt,
    :phasetype => :pt
)

@inline function _hazard_macro_transition(value)
    isnothing(value) && return nothing, nothing
    if value isa Pair
        return value.first, value.second
    elseif value isa NTuple{2}
        return value[1], value[2]
    else
        throw(ArgumentError("`transition` must be a Pair or NTuple{2}"))
    end
end

function _hazard_macro_coalesce(key::Symbol, values...)
    for value in values
        isnothing(value) && continue
        return value
    end
    throw(ArgumentError("@hazard requires `$key = ...`"))
end

function _hazard_macro_entry(; formula = nothing,
                               hazard = nothing,
                               family = nothing,
                               statefrom = nothing,
                               stateto = nothing,
                               from = nothing,
                               to = nothing,
                               transition = nothing,
                               states = nothing,
                               kwargs...)
    hazard_formula = something(formula, hazard, _DEFAULT_HAZARD_FORMULA)

    isnothing(family) && throw(ArgumentError("@hazard requires `family = ...`"))
    family_symbol = family isa Symbol ? family : Symbol(lowercase(String(family)))

    transition_from, transition_to = _hazard_macro_transition(transition)
    states_from, states_to = _hazard_macro_transition(states)

    state_from = _hazard_macro_coalesce(:statefrom, statefrom, from, transition_from, states_from)
    state_to = _hazard_macro_coalesce(:stateto, stateto, to, transition_to, states_to)

    if haskey(_ANALYTIC_HAZARD_FAMILY_ALIASES, family_symbol)
        normalized_family = _ANALYTIC_HAZARD_FAMILY_ALIASES[family_symbol]
        return Hazard(hazard_formula, normalized_family, state_from, state_to; kwargs...)
    elseif family_symbol in (:ode, :neural_ode)
        throw(ArgumentError("ODE-backed hazards are not yet implemented. Please use an analytic family (:exp, :wei, :gom, :sp) for now."))
    else
        throw(ArgumentError("Unknown hazard family `$(family_symbol)`"))
    end
end

function _hazard_macro_kwexprs(body)
    args = body isa Expr && body.head == :block ? body.args : (isnothing(body) ? Expr[] : [body])
    kwexprs = Expr[]
    for arg in args
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :(=)
            key = arg.args[1]
            key isa Symbol || throw(ArgumentError("@hazard assignments must use simple symbols (e.g., `family = :exp`)."))
            push!(kwexprs, Expr(:kw, key, arg.args[2]))
        elseif isnothing(arg)
            continue
        else
            throw(ArgumentError("@hazard expects a block of assignments like `key = value`."))
        end
    end
    return kwexprs
end

"""
    @hazard begin
        family = :exp
        statefrom = 1
        stateto = 2
        linpred_effect = :ph
    end

Declarative front-end for building `Hazard` objects. Supply a block of
`key = value` assignments, specify the family plus transition, and optionally add
`formula = @formula(0 ~ x + z)` when covariates are present. If you skip the formula,
the macro injects the intercept-only `@formula(0 ~ 1)` automatically.

Supported families today are `:exp`, `:wei`, `:gom`, and `:sp` (case-insensitive).
Family tags `:ode` and `:neural_ode` are reserved for upcoming numerical hazard
support and will currently throw a descriptive error.

Optional aliases:
- Use `from`, `to`, `states`, or `transition = 1 => 2` instead of `statefrom`/`stateto`.
- Provide either `formula` or `hazard` if you need to override the generated `StatsModels` term.

Any remaining keywords (e.g., `linpred_effect`, `time_transform`, spline controls)
are forwarded to the underlying `Hazard` constructor.
"""
macro hazard(body)
    kwexprs = _hazard_macro_kwexprs(body)
    isempty(kwexprs) && throw(ArgumentError("@hazard requires at least one assignment"))
    return esc(:(MultistateModels._hazard_macro_entry(; $(kwexprs...))))
end
