module TestFixtures

using DataFrames
using StatsModels
using MultistateModels

export baseline_exact_data,
       baseline_exact_covariates,
       censoring_panel_data,
       duplicate_transition_data,
       noncontiguous_state_data,
       exp_hazard,
       wei_hazard,
       gom_hazard

const INTERCEPT_ONLY = @formula(0 ~ 1)

"""Two-interval panel data without covariates (baseline-only hazards)."""
function baseline_exact_data()
    return DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2]
    )
end

"""Two-interval panel data with covariates `age` and `trt`."""
function baseline_exact_covariates()
    df = baseline_exact_data()
    df.age = [50, 51]
    df.trt = [0, 1]
    return df
end

"""Panel data with a censored interval (stateto = 0) that relies on CensoringPatterns."""
function censoring_panel_data()
    return DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [2, 0],
        obstype = [1, 3]
    )
end

"""Generic three-row panel data used by duplicate-transition tests."""
function duplicate_transition_data()
    return DataFrame(
        id = [1, 1, 1],
        tstart = [0.0, 1.0, 2.0],
        tstop = [1.0, 2.0, 3.0],
        statefrom = [1, 2, 3],
        stateto = [2, 3, 3],
        obstype = [1, 1, 3]
    )
end

"""Data with states 1, 2, and 4 to trigger non-contiguous state validation."""
function noncontiguous_state_data()
    return DataFrame(
        id = [1, 1, 1],
        tstart = [0.0, 1.0, 2.0],
        tstop = [1.0, 2.0, 3.0],
        statefrom = [1, 2, 4],
        stateto = [2, 4, 4],
        obstype = [1, 1, 3]
    )
end

exp_hazard(; formula = INTERCEPT_ONLY, from::Int = 1, to::Int = 2, kwargs...) =
    Hazard(formula, "exp", from, to; kwargs...)

wei_hazard(; formula = INTERCEPT_ONLY, from::Int = 1, to::Int = 2, kwargs...) =
    Hazard(formula, "wei", from, to; kwargs...)

gom_hazard(; formula = INTERCEPT_ONLY, from::Int = 1, to::Int = 2, kwargs...) =
    Hazard(formula, "gom", from, to; kwargs...)

end # module
