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
        if !any((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
            knots = copy(timespan)
        else
            knots = [0.0, maximum(extract_sojourns(hazard, data, extract_paths(data; self_transitions = false); sojourns_only = true))]
        end
    else
        if !any((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
            sojourns = copy(timespan)
        else
            sojourns = [0.0, maximum(extract_sojourns(hazard, data, extract_paths(data; self_transitions = false); sojourns_only = true))]
        end

        knots = hazard.knots
        ra = "→"; sf = hazard.statefrom; st = hazard.stateto

        if any((knots .< timespan[1]) .| (knots .> timespan[2]))
            @warn "A knot for the $sf $ra $st transition was set outside of the range of time intervals in the data."
        end
        
        if length(knots) == 1
            @warn "A single knot location is specified for the $sf $ra $st transition, it will be assumed to be an interior knot. The boundaries have been set at 0 and the maximum observed sojourn."
            
            knots = sort([knots; timespan])

        elseif hazard.add_boundaries
            @info "Boundary knot locations have been added for the $sf $ra $st transition. The boundaries have been set at 0 and the maximum observed sojourn."
            
            knots = sort([knots; timespan])
        end
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
    if !(hazard.extrapolation ∈ ["flat", "linear"])
        @error "The extrapolation method for splines must be one of `flat` or `linear`"
    end

    extrap_method = hazard.extrapolation == "linear" ? BSplineKit.Linear() : BSplineKit.Flat()
    sphaz = SplineExtrapolation(Spline(undef, B), extrap_method)
    spchaz = SplineExtrapolation(integral(sphaz), extrap_method)

    return (sphaz = sphaz, spchaz = spchaz, rmat = M, knots = knots, timespan = timespan)
end

"""
    recombine_parameters!(hazard::_SplineHazard, parameters)

Copy recombined parameters to the spline objects for hazard and cumulative hazard.
"""
function recombine_parameters!(hazard::_SplineHazard, parameters)
    
    # write B-spline recombined B-spline parameters
    copyto!(hazard.hazsp.spline.coefs, hazard.rmat * exp.(parameters[1:size(hazard.rmat, 2)]))

    # get spline specs
    t = knots(hazard.hazsp)
    k = BSplineKit.order(hazard.hazsp)

    # initialize first coefficient at
    hazard.chazsp.spline.coefs[begin] = zero(eltype(hazard.chazsp.spline.coefs))

    # write recombined parameters for integral
    @inbounds for i in eachindex(hazard.hazsp.spline.coefs)
        hazard.chazsp.spline.coefs[i + 1] = 
            hazard.chazsp.spline.coefs[i] + hazard.hazsp.spline.coefs[i] * (t[i + k] - t[i]) / k
    end
end

"""
    set_riskperiod!(hazard::_SplineHazard)

Calculate and set the risk period for when a spline intensity is greater than zero. 
"""
function set_riskperiod!(hazard::_SplineHazard)

    if hazard.hazsp.method == Flat()
        copyto!(hazard.riskperiod, hazard.timespan)
    else
        # get spline boundaries
        sp_bounds = boundaries(hazard.hazsp.spline.basis)
        
        # initialize diffresult object
        riskstart = DiffResults.DiffResult(0.0, (0.0,))
        riskend = DiffResults.DiffResult(0.0, (0.0,))

        # compute value and slope at endpoints
        riskstart = ForwardDiff.derivative!(riskstart, hazard.hazsp, sp_bounds[1])
        riskend = ForwardDiff.derivative!(riskend, hazard.hazsp, sp_bounds[2])

        # set riskperiod
        if riskstart.derivs[1] <= 0.0
            hazard.riskperiod[1] = hazard.timespan[1]
        else
            hazard.riskperiod[1] = maximum([hazard.timespan[1], sp_bounds[1] - riskstart.value / riskstart.derivs[1]])
        end

        # set riskperiod
        if riskend.derivs[1] >= 0.0
            hazard.riskperiod[2] = hazard.timespan[2]
        else
            hazard.riskperiod[2] = minimum([hazard.timespan[2], sp_bounds[2] - riskend.value / riskend.derivs[1]])
        end
    end

end