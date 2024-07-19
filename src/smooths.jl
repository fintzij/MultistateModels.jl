"""
    spline_hazards(hazard::SplineHazard, data::DataFrame)

Spline hazard object generated via BSplineKit.jl. 

Arguments:

    - hazard: a SplineHazard object
    - data: DataFrame
"""
function spline_hazards(hazard::SplineHazard, data::DataFrame)

    # boundary knots
    timespan = [minimum(data.tstart), maximum(data.tstop)]

    # interior knots and df
    # if no transitions observed (e.g., initializing a dataset) then use timespan
    if !any((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
        boundaries = copy(timespan)
    else
        # calculate maximum sojourn
        boundaries = [0.0, maximum(extract_sojourns(hazard, data, extract_paths(data; self_transitions = false); sojourns_only = true))]
    end

    knots = hazard.knots
    ra = "→"; sf = hazard.statefrom; st = hazard.stateto

    if any((knots .< timespan[1]) .| (knots .> timespan[2]))
        @warn "A knot for the $sf $ra $st transition was set outside of the time span of the data."
    end

    if any((knots .< boundaries[1]) .| (knots .> boundaries[2]))
        @warn "A knot for the $sf $ra $st transition was set outside of the sojourns observed in the data."
    end

    # generate Splines
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(knots))

    if (hazard.degree > 1) & hazard.natural_spline
        # recombine
        B = RecombinedBSplineBasis(B, Derivative(2))
        M = Matrix{Float64}(B.M)

        # check if the spline basis is large enough for the recombination        
        testspline = approximate(x -> exp(1), B)

        if !all((M * coefficients(testspline)) .≈ mean(M * coefficients(testspline)))
            @error "The spline basis is not large enough to support the recombination procedure for natural splines. Consider using polynomials of another degree, adding knots, or an unrestricted spline."
        end

    else
        M = diagm(ones(Float64, length(B)))
    end
    
    # generate splines with extrapolations
    if !(hazard.extrapolation ∈ ["flat", "linear"])
        @error "The extrapolation method for splines must be one of `flat` or `linear`"
    end

    extrap_method = hazard.extrapolation == "linear" ? BSplineKit.Linear() : BSplineKit.Flat()
    sphaz = SplineExtrapolation(Spline(undef, B), extrap_method)
    spchaz = integral(sphaz.spline)

    return (sphaz = sphaz, spchaz = spchaz, rmat = M, knots = knots, timespan = timespan)
end

"""
    remake_splines!(hazard::_SplineHazard, parameters)

Remake splines in hazard object with new parameters.
"""
function remake_splines!(hazard::_SplineHazard, parameters)
    
    # make new spline
    hazsp = Spline(hazard.hazsp.spline.basis, exp.(parameters[1:hazard.nbasis]))
    chazsp = integral(hazsp)

    # remake spline objects with recombined parameters
    hazard.hazsp = SplineExtrapolation(hazsp, hazard.hazsp.method);
    hazard.chazsp = chazsp;
end

"""
    set_riskperiod!(hazard::_SplineHazard)

Calculate and set the risk period for when a spline intensity is greater than zero. 
"""
function set_riskperiod!(hazard::_SplineHazard)

    if hazard.hazsp.method == BSplineKit.Linear()
        # get spline boundaries
        sp_bounds = boundaries(hazard.hazsp.spline.basis)
        
        # spline derivative
        D = BSplineKit.diff(hazard.hazsp.spline)

        # compute derivatives
        spvalues = hazard.hazsp.spline.(sp_bounds)
        spderivs = ForwardDiff.derivative.(hazard.hazsp.spline, [sp_bounds...])

        # set riskperiod
        riskperiod = [spderivs[1] <= 0 ? hazard.timespan[1] : maximum([hazard.timespan[1], sp_bounds[1] - spvalues[1] / spderivs[1]]),
                      spderivs[2] >= 0 ? hazard.timespan[2] : minimum([hazard.timespan[2], sp_bounds[2] - spvalues[2] / spderivs[2]])]

        # copy
        hazard.riskperiod = riskperiod
    end
end

