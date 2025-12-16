"""
Utility functions for statistical transforms shared across diagnostics, tests,
and core likelihood components.
"""

"""
    truncate_distribution(cdf_base, pdf_base; lower = 0.0, upper = Inf, atol = 1e-12)

Normalize `cdf_base`/`pdf_base` over the interval `(lower, upper)` and return a
pair of closures representing the truncated distribution. The helper enforces a
strictly positive probability mass on the requested window and clamps values
outside the bounds to `0` or `1` as appropriate.
"""
function truncate_distribution(cdf_base::Function, pdf_base::Function;
                               lower::Real = 0.0,
                               upper::Real = Inf,
                               atol::Real = 1e-12)
    lower < upper || throw(ArgumentError("lower must be < upper (got $(lower), $(upper))"))

    cdf_lower = isfinite(lower) ? cdf_base(lower) : zero(Float64)
    cdf_upper = isfinite(upper) ? cdf_base(upper) : one(Float64)
    mass = cdf_upper - cdf_lower
    mass > atol || throw(ArgumentError("Truncation interval ($(lower), $(upper)) has insufficient probability mass ($(mass))."))

    function cdf_trunc(t)
        if t <= lower
            return 0.0
        elseif t >= upper
            return 1.0
        else
            return (cdf_base(t) - cdf_lower) / mass
        end
    end

    function pdf_trunc(t)
        if t <= lower || t >= upper
            return 0.0
        else
            return pdf_base(t) / mass
        end
    end

    return cdf_trunc, pdf_trunc
end
