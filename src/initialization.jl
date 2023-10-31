"""
    set_crude_init!(model::MultistateProcess)

Modify the parameter values in a MultistateProcess object
"""
function set_crude_init!(model::MultistateProcess)
    crude_par = calculate_crude(model)
    for i in model.hazards
        set_par_to = init_par(i, log(crude_par[i.statefrom, i.stateto]))
        set_parameters!(model, NamedTuple{(i.hazname,)}((set_par_to,)))
        # if !isa(i, _Spline)
        #     set_par_to = init_par(i, log(crude_par[i.statefrom, i.stateto]))
        #     set_parameters!(model, NamedTuple{(i.hazname,)}((set_par_to,)))
        # else
        #     @warn "Parameters for $(i.hazname) not initialized by set_crude_init!(). Parameters for spline hazards should be initialized manually using set_parameters!()"
        # end
    end
end

"""
    compute_suff_stats(dat, tmat)

Return a matrix in same format as tmat with observed transition counts, and a vector of time spent in each state. Used for checking data and calculating crude initialization rates.
"""
function compute_suff_stats(data, tmat)
    # matrix to store number of transitions from state r to state s, stored using same convention as model.tmat
    n_rs = zeros(size(tmat))

    # vector of length equal to number of states to store time spent in state
    T_r = zeros(size(tmat)[1])

    for rd in eachrow(data)
        # loop through data rows
        
        # track how much time has been spent in state
        #T_r_accumulator += r.tstop - r.tstart
        if rd.statefrom>0
            T_r[rd.statefrom] += rd.tstop - rd.tstart
        end        
        # if a state transition happens then increment transition by 1 in n_rs
        if rd.statefrom != rd.stateto
            if rd.statefrom>0 && rd.stateto>0
                n_rs[rd.statefrom, rd.stateto] += 1
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
    n_rs, T_r = compute_suff_stats(model.data, model.tmat)

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


#exp_rates = MultistateModels.calculate_crude(model)
#map(x -> match_moment(x, exp_rates[x.statefrom, x.stateto]))

#output = similar(msm_2state_transadj.parameters)
#copyto!(output, map(x -> MultistateModels.match_moment(x, cmat[x.statefrom, x.stateto]), msm_2state_transadj.hazards))


# COMPARE calculate_crude() to fit_exact()
# SHOULD WE PASS PARAMETERS ON LOG SCALE OR NATIVE SCALE
# MATCHING MOMENTS WITH GOMPERTZ
# PREALLOCATE VECTOR OF VECTORS TO STORE PARAMETERS?

"""

Pass-through the crude exponential rate. Method for exponential hazard with no covariates.
"""
function init_par(_hazard::_Exponential, crude_log_rate=0)
    return [crude_log_rate]
end


"""

Pass-through the crude exponential rate and zero out the coefficients for covariates. Method for exponential hazard with covariates.
"""
function init_par(_hazard::_ExponentialPH, crude_log_rate=0)
    return vcat(crude_log_rate, zeros(size(_hazard.data, 2) - 1))
end


"""

Weibull without covariates
"""
function init_par(_hazard::_Weibull, crude_log_rate=0)
    # set shape parameter to 0 (i.e. log(1)) so it's exponential
    # pass through crude log exponential rate
    return [0, crude_log_rate]
end

"""

Weibull with covariates
"""
function init_par(_hazard::_WeibullPH, crude_log_rate=0)
    # set shape parameter to 0 (i.e. log(1)) so it's exponential
    # pass through crude exponential rate
    # set covariate coefficients to 0
    return vcat([0, crude_log_rate], zeros(size(_hazard.data, 2) - 1))
end


"""

Gompertz without covariates
"""
function init_par(_hazard::_Gompertz, crude_log_rate=0)
    # set shape to 0 (i.e. log(1)) 
    # pass through crude exponential rate 
    return [0, crude_log_rate]
end

"""

Gompertz with covariates
"""
function init_par(_hazard::_GompertzPH, crude_log_rate=0)
    # set shape to 0 (i.e. log(1)) 
    # pass through crude exponential rate 
    # set covariate coefficients to 0
    return vcat([0, crude_log_rate], zeros(size(_hazard.data, 2) - 1))
end


"""

spline without covariates
"""
function init_par(_hazard::_Spline, crude_log_rate=0)
    # set shape to 0 (i.e. log(1)) 
    # pass through crude exponential rate
    npars = length(_hazard.parnames)
    return repeat([crude_log_rate], npars)
end

"""

spline with covariates
"""
function init_par(_hazard::_SplinePH, crude_log_rate=0)
    # set shape to 0 (i.e. log(1)) 
    # pass through crude exponential rate 
    # set covariate coefficients to 0
    return vcat(repeat([crude_log_rate], length(_hazard.parnames) - size(_hazard.data, 2)), zeros(size(_hazard.data, 2) - 1))
end