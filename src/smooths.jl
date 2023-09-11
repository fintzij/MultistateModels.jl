"""
    spline_hazards(hazard::SplineHazard, data::DataFrame, samplepaths::Vector{SamplePath})

Get spline hazard objects via splines2. 
"""
function spline_hazards(hazard::SplineHazard, data::DataFrame, samplepaths::Vector{SamplePath})

    # observed times
    sojourns = unique(sort([minimum(data.tstart); extract_sojourns(hazard, data, samplepaths; sojourns_only = true); maximum(data.tstop)]))

    ## unpack arguments
    # degrees of freedom
    df = isnothing(hazard.df) ? 0 : hazard.df

    # boundary knots
    boundaryknots = isnothing(hazard.boundaryknots) ? [minimum(data.tstart), maximum(data.tstop)] : hazard.boundaryknots

    # interior knots
    knots = isnothing(hazard.knots) ? Float64[] : hazard.knots

    # intercept, periodic, and degree
    intercept = true
    periodic = hazard.periodic
    degree = hazard.degree

    # get spline objects from splines2
    if !hazard.monotonic
        # mSpline via splines2
        sphaz = R"t(splines2::mSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))"

        # iSpline via splines2
        spchaz = R"t(splines2::iSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))"
    else
        # mSpline via splines2
        sphaz = R"t(splines2::iSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))"

        # iSpline via splines2
        spchaz = R"t(splines2::cSpline($sojourns, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))"
    end

    return (hazard = sphaz, cumulative_hazard = spchaz, times = sojourns)
end

"""
    spline_setup!(hazard::_Spline, data::DataFrame, samplepaths::Vector{SamplePath})

Set up the spline basis for a semi-parametric hazard function given a set of sample paths.
"""
function compute_spline_basis!(hazard::Union{_Spline, _SplinePH}, data::DataFrame, samplepaths::Vector{SamplePath})

    # get new times
    times = unique(setdiff(extract_sojourns(hazard, data, samplepaths; sojourns_only = false), hazard.times))
 
    # compute new basis and append 
    if length(times) != 0
        append!(hazard.times, times)
        append!(hazard.hazbasis, rcopy(R"t(predict($(hazard.hazobj), newx = $times))"))
        append!(hazard.chazbasis, rcopy(R"t(predict($(hazard.chazobj), newx = $times))"))

        # sort
        inds = sortperm(hazard.times)
        permute!(hazard.times, inds)
        Base.permutecols!!(hazard.hazbasis, copy(inds))
        Base.permutecols!!(hazard.chazbasis, inds)
    end
end
