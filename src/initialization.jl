"""
    set_crude_init!(model::MultistateProcess)

Modify the parameter values in a MultistateProcess object
"""
function set_crude_init!(model::MultistateProcess; constraints = nothing)

    if !isnothing(constraints)
        error("Cannot initialize parameters to crude estimates when there are parameter constraints.")
    
    elseif isnothing(constraints)
        
        crude_par = calculate_crude(model)

        for i in model.hazards
            set_par_to = init_par(i, log(crude_par[i.statefrom, i.stateto]))
            set_parameters!(model, NamedTuple{(i.hazname,)}((set_par_to,)))
        end

        for i in eachindex(model.hazards)
            if isa(model.hazards[i], _SplineHazard)
                log_params = get_log_scale_params(model.parameters)
                remake_splines!(model.hazards[i], log_params[i])
                set_riskperiod!(model.hazards[i])
            end
        end
    end
end

"""
    initialize_parameters(model::MultistateProcess; constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing, crude = false)

Modify the parameter values in a MultistateProcess object, calibrate to the MLE of a Markov surrogate.
"""
function initialize_parameters(model::MultistateProcess; constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing, crude = false)

    model = deepcopy(model)

    if crude
        set_crude_init!(model; constraints = constraints)
    else
        # check that surrogate constraints are supplied if there are other constraints
        if !isnothing(constraints) && isnothing(surrogate_constraints)
            @error "Constraints for the Markov surrogate must be provided if there are constraints on the model parameters."
        end

        # fit Markov surrogate
        surrog = fit_surrogate(model; surrogate_constraints = constraints, surrogate_parameters = surrogate_parameters, verbose = false)

        for i in eachindex(model.hazards)
            set_par_to = init_par(model.hazards[i], surrog.parameters[i][1])

            # copy covariate effects if there are any
            if typeof(model.hazards[i]) ∈ [_ExponentialPH, _WeibullPH, _GompertzPH, _SplinePH]
                set_par_to[reverse(range(length(set_par_to); step = -1, length = model.hazards[i].ncovar))] .= surrog.parameters[i][Not(1)]
            end
            
            set_parameters!(model, NamedTuple{(model.hazards[i].hazname,)}((set_par_to,)))

            if isa(model.hazards[i], _SplineHazard)
                log_params = get_log_scale_params(model.parameters)
                remake_splines!(model.hazards[i], log_params[i])
                set_riskperiod!(model.hazards[i])
            end
        end
    end

    return model
end

"""
    initialize_parameters!(model::MultistateProcess; constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing, crude = false)

Modify the parameter values in a MultistateProcess object, calibrate to the MLE of a Markov surrogate.
"""
function initialize_parameters!(model::MultistateProcess; constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing, crude = false)

    if crude
        set_crude_init!(model; constraints = constraints)
    else
        # check that surrogate constraints are supplied if there are other constraints
        if !isnothing(constraints) && isnothing(surrogate_constraints)
            @error "Constraints for the Markov surrogate must be provided if there are constraints on the model parameters."
        end

        # fit Markov surrogate
        surrog = fit_surrogate(model; surrogate_constraints = constraints, surrogate_parameters = surrogate_parameters, verbose = false)

        for i in eachindex(model.hazards)
            set_par_to = init_par(model.hazards[i], surrog.parameters[i][1])

            # copy covariate effects if there are any
            if typeof(model.hazards[i]) ∈ [_ExponentialPH, _WeibullPH, _GompertzPH, _SplinePH]
                set_par_to[reverse(range(length(set_par_to); step = -1, length = model.hazards[i].ncovar))] .= surrog.parameters[i][Not(1)]
            end
            
            set_parameters!(model, NamedTuple{(model.hazards[i].hazname,)}((set_par_to,)))

            if isa(model.hazards[i], _SplineHazard)
                log_params = get_log_scale_params(model.parameters)
                remake_splines!(model.hazards[i], log_params[i])
                set_riskperiod!(model.hazards[i])
            end
        end
    end
end

"""
    compute_suff_stats(dat, tmat, SubjectWeights)

Return a matrix in same format as tmat with observed transition counts, and a vector of time spent in each state. Used for checking data and calculating crude initialization rates.
"""
function compute_suff_stats(data, tmat, SubjectWeights)
    # matrix to store number of transitions from state r to state s, stored using same convention as model.tmat
    n_rs = zeros(size(tmat))

    # vector of length equal to number of states to store time spent in state
    T_r = zeros(size(tmat)[1])

    for rd in eachrow(data)
        # loop through data rows
        # track how much time has been spent in state
        #T_r_accumulator += r.tstop - r.tstart
        if rd.statefrom>0
            T_r[rd.statefrom] += (rd.tstop - rd.tstart) * SubjectWeights[rd.id]
        end        
        # if a state transition happens then increment transition by 1 in n_rs
        if rd.statefrom != rd.stateto
            if rd.statefrom>0 && rd.stateto>0
                n_rs[rd.statefrom, rd.stateto] += SubjectWeights[rd.id]
            end
        end
    end

    return n_rs, T_r
end

"""
    compute_number_transitions(data, tmat)

Return a matrix in same format as tmat with observed transition counts, and a vector of time spent in each state. Used for checking data and calculating crude initialization rates.
"""
function compute_number_transitions(data, tmat)
    # matrix to store number of transitions from state r to state s, stored using same convention as model.tmat
    n_rs = zeros(size(tmat))

    for rd in eachrow(data)
        # loop through data rows
        if rd.statefrom != rd.stateto
            # if a state transition happens then increment transition by 1 in n_rs
            if rd.statefrom>0 && rd.stateto>0
                n_rs[rd.statefrom, rd.stateto] += 1
            end
        end
    end

    return n_rs
end

# Raphael comment: what about progressive states. E.g. say you can go from 1 to 2 and 2 to 3, but directly from 1 to 3 is NOT allowed. But we have panel data and observe a 1 to 3 transition - that implies that someone definitely spent time in 2x, and as the above is currently written, we would accrue a 1 to 3 transition which is not allowed. 2023 June 6th.

# Also, for panel crude inits maybe split time between states instead of put all in initial state

# What about e.g. cancer case where someone is diagnosed at age 45, but realistically they probably have been in that "state" since 30. If fitting Markov model, who cares. But if Weibull or semi-markov, would want to make sure they are 15 years into the hazard function.

"""
    calculate_crude(model::MultistateProcess)

Return a matrix with same dimensions as model.tmat, but row i column j entry is number of transitions from state i to state j divided by time spent in state i, then log transformed. In other words, a faster version of log exponential rates that fit_exact would return.

Accept a MultistateProcess object.
"""
function calculate_crude(model::MultistateProcess)
    # n_rs is matrix like tmat except each entry is number of transitions from state r to s
    # T_r is vector of length number of states
    n_rs, T_r = compute_suff_stats(model.data, model.tmat, model.SubjectWeights)

    # crude fix to avoid taking the log of zero (for pairs of states with no observed transitions) by turning zeros to 0.5. Also check_data!() should have thrown an error during model generation if this is the case.
    n_rs = max.(n_rs, 0.5)
    
    # give a reasonable sojourn time to states never visited
    T_r[T_r .== 0] .= mean(model.data.tstop - model.data.tstart)

# UNALLOWED TRANSITIONS SHOULD BE ZERO
# ALLOWED TRANSITIONS WITH ZERO SHOULD BE 0.5
# DIAGONALS SHOULD BE ROW SUMS, WHICH INCLUDE ANY 0.5

    # return log of the rate
    crude_mat = n_rs ./ T_r
    
    crude_mat[findall(model.tmat .== 0)] .= 0

    for i in 1:length(T_r)
        crude_mat[i, i] =  -sum(crude_mat[i, Not(i)])
    end
    
    return crude_mat
    #give_log ? log.(crude_mat) : crude_mat 
end

"""
Initialize parameters for new hazard types (MarkovHazard, SemiMarkovHazard, RuntimeSplineHazard).

Dispatches based on family and has_covariates to determine parameter initialization.
"""
function init_par(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, crude_log_rate=0.0)
    family = hazard.family
    has_covs = hazard.has_covariates
    ncovar = hazard.npar_total - hazard.npar_baseline
    
    if family == "exp"
        # Exponential: [log_baseline] or [log_baseline, β1, β2, ...]
        return has_covs ? vcat(crude_log_rate, zeros(ncovar)) : [crude_log_rate]
        
    elseif family == "wei"
        # Weibull: [log_shape, log_scale] or [log_shape, log_scale, β1, β2, ...]
        # Initialize shape=1 (log_shape=0) to start as exponential
        baseline = [0.0, crude_log_rate]  # log(shape=1), log_scale
        return has_covs ? vcat(baseline, zeros(ncovar)) : baseline
        
    elseif family == "gom"
        # Gompertz: [log_shape, log_scale] or [log_shape, log_scale, β1, β2, ...]
        # Initialize shape=1 (log_shape=0) to start as exponential
        baseline = [0.0, crude_log_rate]  # log(shape=1), log_scale
        return has_covs ? vcat(baseline, zeros(ncovar)) : baseline
        
    elseif family == "sp"
        # Spline: Initialize all spline coefficients to give constant hazard
        # log(coef) = 0 → coef = 1 → constant hazard at value 1
        # The crude_log_rate can be used to shift the overall level
        nbasis = hazard.npar_baseline
        baseline = fill(crude_log_rate, nbasis)
        return has_covs ? vcat(baseline, zeros(ncovar)) : baseline
        
    else
        error("Unknown hazard family: $family")
    end
end
