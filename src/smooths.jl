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
    if isnothing(hazard.knots)
        # if no transitions observed (e.g., initializing a dataset) then use timespan
        if !any((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
            knots = copy(timespan)
        else
            # else use 0 and maximum sojourns
            knots = sort(unique([0.0, maximum(extract_sojourns(hazard, data, extract_paths(data; self_transitions = false); sojourns_only = true))]))
        end
    else
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
        
        if length(knots) == 1
            @warn "A single knot location is specified for the $sf $ra $st transition, it will be assumed to be an interior knot. The boundaries have been set at 0 and $(boundaries[2])."
            
            knots = sort(unique([knots; boundaries]))

        elseif hazard.add_boundaries
            @info "Boundary knot locations have been added for the $sf $ra $st transition. The boundaries have been set at 0 and $(boundaries[2])."
            
            knots = sort(unique([knots; boundaries]))
        end
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
    spchaz = SplineExtrapolation(integral(sphaz.spline), extrap_method)

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
    hazard.hazsp = SplineExtrapolation(hazsp, hazard.hazsp.method)
    hazard.chazsp = SplineExtrapolation(chazsp, hazard.chazsp.method)
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
        spvalues = ForwardDiff.value.(hazard.hazsp.spline.(sp_bounds))
        spderivs = ForwardDiff.value.(D.(sp_bounds))

        # set riskperiod
        riskperiod = zeros(Float64, 2)

        if spderivs[1] > 0.0
            riskperiod[1] = maximum([hazard.timespan[1], sp_bounds[1] - ForwardDiff.value(spvalues[1]) / ForwardDiff.value(spderivs[1])])
        else
            riskperiod[1] = hazard.timespan[1]
        end

        # set riskperiod
        if spderivs[2] < 0.0
            riskperiod[2] = minimum([hazard.timespan[2], sp_bounds[2] - ForwardDiff.value(spvalues[2]) / ForwardDiff.value(spderivs[2])])
        else
            riskperiod[2] = hazard.timespan[2]
        end

        # copy
        copyto!(hazard.riskperiod, riskperiod)
    end
end

