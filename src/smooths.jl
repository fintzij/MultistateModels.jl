"""
    spline_hazards(hazard::SplineHazard, data::DataFrame)

Spline hazard object generated via BSplineKit.jl. 
"""
function spline_hazards(hazard::SplineHazard, data::DataFrame)

    # boundary knots
    timespan = [minimum(data.tstart), maximum(data.tstop)]
    riskperiod = copy(timespan)

    # interior knots and df
    if isnothing(hazard.knots)
        if !any((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
            knots = copy(timespan)
        else
            knots = [0.0, maximum(extract_sojourns(hazard, data, extract_paths(data; self_transitions = false); sojourns_only = true))]
        end
    else
        knots = hazard.knots
    end

    # generate Splines
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(knots))

    if hazard.degree > 1
        # recombine
        R = RecombinedBSplineBasis(B, Derivative(2))
        M = Matrix{Float64}(R.M)
    else
        M = diagm(ones(Float64, length(B)))
    end
    
    # generate splines with extrapolations
    sphaz = SplineExtrapolation(Spline(undef, B), Linear())
    spchaz = SplineExtrapolation(integral(sphaz), Linear())

    return (sphaz = sphaz, spchaz = spchaz, rmat = M, knots = knots, riskperiod, timespan = timespan)
end
