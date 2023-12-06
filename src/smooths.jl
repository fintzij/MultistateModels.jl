"""
    spline_hazards(hazard::SplineHazard, data::DataFrame)

Get spline hazard objects via splines2. 
"""
function spline_hazards(hazard::SplineHazard, data::DataFrame)

    # observed times
    mesh = collect(LinRange(minimum(data.tstart), maximum(data.tstop), hazard.meshsize))

    ## unpack arguments
    # intercept, periodic, and degree
    intercept = true
    periodic = hazard.periodic
    degree = hazard.degree

    # boundary knots
    boundaryknots = isnothing(hazard.boundaryknots) ? [minimum(data.tstart), maximum(data.tstop)] : hazard.boundaryknots

    # interior knots and df
    if isnothing(hazard.knots)
        # get degrees of freedom
        df = isnothing(hazard.df) ? degree + 1 : hazard.df
        nknots = df - degree - 1

        # get knots
        if nknots < 0
            sfrom = hazard.statefrom; sto = hazard.stateto
            @error "The spline for the transition from state $sfrom to $sto has too few degrees of freedom." 
        elseif nknots == 0
            knots = Float64[]
        else
            sojourns = unique(sort([minimum(data.tstart); extract_sojourns(hazard, data, samplepaths; sojourns_only = false); maximum(data.tstop)]))
            knots = quantile(sojourns, collect(1:nknots) / (nknots + 1))
        end

    else
        knots = hazard.knots
        df = isnothing(hazard.df) ? length(knots) + degree + 1 : hazard.df
    end

    # get spline objects from splines2
    if !hazard.monotonic
        # mSpline via splines2
        sphaz = rcopy(Array{Float64}, R"t(splines2::mSpline($mesh, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))")

        # iSpline via splines2
        spchaz = rcopy(Array{Float64}, R"t(splines2::iSpline($mesh, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))")
    else
        # mSpline via splines2
        sphaz = rcopy(Array{Float64}, R"t(splines2::iSpline($mesh, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))")

        # iSpline via splines2
        spchaz = rcopy(Array{Float64}, R"t(splines2::cSpline($mesh, df = $df, knots = $knots, degree = $degree, intercept = $intercept, Boundary.knots = $boundaryknots, periodic = $periodic))")
    end

    return (hazard = sphaz, cumulative_hazard = spchaz)
end
