"""
    spline_hazards(hazard::SplineHazard, data::DataFrame)

Spline hazard object generated via BSplineKit.jl. 

Arguments:

    - hazard: a SplineHazard object
    - data: DataFrame
"""
function spline_hazards(hazard::SplineHazard, data::DataFrame)

    # transition
    ra = "→"; sf = hazard.statefrom; st = hazard.stateto

    # boundary knots
    timespan = [minimum(data.tstart), maximum(data.tstop)]

    # if no transitions observed (e.g., initializing a dataset) then use timespan
    if !any((data.statefrom .== hazard.statefrom) .& (data.stateto .== hazard.stateto))
        spbounds = [0.0, timespan[2]]
    else
        # calculate maximum sojourn
        spbounds = [0.0, maximum([extract_sojourns(hazard.statefrom, hazard.stateto, extract_paths(data; self_transitions = false)); extract_sojourns(hazard.statefrom, hazard.statefrom, extract_paths(data; self_transitions = false))])]
    end

    # grab boundary knots
    bknots = isnothing(hazard.boundaryknots) ? spbounds : hazard.boundaryknots

    if ((bknots[1] > spbounds[1]) & (hazard.extrapolation == "linear"))
        @warn "The left boundary for the $sf $ra $st transition was set above the minimum possible sojourn and will be set to 0 because extrapolation method is linear."
        bknots[1] = spbounds[1]
    end

    if ((bknots[2] < spbounds[2]) & (hazard.extrapolation == "linear"))
        @warn "The right boundary for the $sf $ra $st transition was set before the greatest sojourn observed in the data and will be set to $(spbounds[2]) because the extrapolation method is linear."
        bknots[2] = spbounds[2]
    end

    # snag interior knots
    intknots = isnothing(hazard.knots) ? Vector{Float64}() : hazard.knots

    # warn if any interior knots are outside of the boundaries
    if !isnothing(hazard.knots) && (any(intknots .< bknots[1]) | any(intknots .> bknots[2]))
        @warn "Interior knots were specified outside of the spline boundary knots. The boundary knots will be set to the range of the interior knots."
        bknots[1] = bknots[1] > minimum(intknots) ? minimum(intknots) : bknots[1]
        bknots[2] = bknots[2] < maximum(intknots) ? maximum(intknots) : bknots[2]
    end

    # make knots
    spknots = unique(sort([bknots[1]; intknots; bknots[2]]))    

    # generate Splines
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(spknots))

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

    extrap_method = hazard.extrapolation == "linear" ? BSplineKit.SplineExtrapolations.Linear() : BSplineKit.SplineExtrapolations.Flat()
    sphaz = SplineExtrapolation(Spline(undef, B), extrap_method)
    spchaz = integral(sphaz.spline)

    return (sphaz = sphaz, spchaz = spchaz, rmat = M, knots = spknots, timespan = timespan)
end



"""
    spline_ests2coefs(coefs; monotone = 0)

Transform spline parameter estimates on their unrestricted estimation scale to coefficients.
"""
function spline_ests2coefs(ests; monotone = 0)

    # transform
    coefs = (monotone == 0) ? exp.(ests) : (monotone == 1) ? cumsum(exp.(ests)) : reverse(cumsum(exp.(ests)))
    
    # clamp numerical zeros
    coefs[findall(isapprox.(coefs, 0.0; atol = sqrt(eps())))] .= zero(eltype(coefs))    

    return coefs
end

"""
    spline_coefs2ests(ests; monotone = 0)

Transform spline coefficients to unrestrected estimation scale parameters.
"""
function spline_coefs2ests(coefs; monotone = 0)

    if monotone == 1
        # get differences
        coefs = [coefs[1]; diff(coefs)]

    elseif monotone == -1
        # get differences in reverse
        coefs = [coefs[end]; diff(reverse(coefs))]
    end

    # clamp numerical errors to zero
    coefs[findall(isapprox.(coefs, 0.0; atol = sqrt(eps())))] .= zero(eltype(coefs))

    ests = log.(coefs)

    return ests
end

"""
    rectify_coefs(ests, monotone)

Pass model estimates through spline coefficient transformations to remove numerical zeros. 
"""
function rectify_coefs!(ests, model)

    nested = VectorOfVectors(ests, model.parameters.elem_ptr)

    for i in eachindex(model.hazards)
        if isa(model.hazards[i], _SplineHazard)
            # get rectified parameters
            rectified = [spline_coefs2ests(spline_ests2coefs(nested[i]; monotone = model.hazards[i].monotone); monotone = model.hazards[i].monotone); nested[i][Not(1:model.hazards[i].nbasis)]]

            # copy back to ests
            deepsetindex!(nested, rectified, i)
        end
    end
end

"""
    remake_splines!(hazard::_SplineHazard, parameters)

Remake splines in hazard object with new parameters.
"""
function remake_splines!(hazard::_SplineHazard, parameters)
    
    # make new spline
    hazsp = Spline(hazard.hazsp.spline.basis, spline_ests2coefs(parameters[1:hazard.nbasis]; monotone = hazard.monotone))
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

    if hazard.hazsp.method == BSplineKit.SplineExtrapolations.Linear()
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

        # clamp and copy
        clamp!(riskperiod, hazard.timespan[1], hazard.timespan[2])
        copyto!(hazard.riskperiod, riskperiod)
    end
end
