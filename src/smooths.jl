"""
    spline_hazards(hazard::SplineHazard, data::DataFrame)

Get spline hazard objects via splines2. 
"""
function spline_hazards(hazard::SplineHazard, data::DataFrame)

    # observed sojourns
    inds = findall((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
    sojourns = unique(data.tstop[inds] - data.tstart[inds])

    ## unpack arguments
    # degrees of freedom
    df = isnothing(hazard.df) ? 0 : hazard.df

    # boundary knots
    boundaryknots = isnothing(hazard.boundaryknots) ? [minimum(data.tstart), maximum(data.tstop)] : hazard.boundaryknots

    # interior knots
    knots = isnothing(hazard.knots) ? Float64[] : hazard.knots

    # intercept, periodic, and degree
    (;intercept,periodic,degree) = hazard

    # get spline objects from splines2
    if hazard.monotonic
        # mSpline via splines2
        sphaz = R"splines2::mSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic)"

        # iSpline via splines2
        spchaz = R"splines2::iSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic)"
    else
        # mSpline via splines2
        sphaz = R"splines2::iSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic)"

        # iSpline via splines2
        spchaz = R"splines2::cSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic)"
    end

    return (hazard = sphaz, cumulative_hazard = spchaz)
end