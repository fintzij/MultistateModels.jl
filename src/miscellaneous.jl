# store parameters in tuple or vector of vectors
# collect time spent in each state
#  store in vector of time in states T_r (msm R notation)
# collect transitions
#  store in matrix like tmat, refactor compute_statetable() and statetable()

# loop through hazards in model
# loop through data to collect events/time at risk

#m2 = deepcopy(model)
#initpar = deepcopy(m2.parameters)
#T_r = zeros(size(m2.tmat)[1])
#n_rs = zeros(size(m2.tmat))

function calculate_crude(model::MultistateModel)
    usubj = unique(model.data.id)
    T_r = zeros(size(model.tmat)[1])
    n_rs = zeros(size(model.tmat))

    for s in usubj
        # loop through all unique subjects
        
        # container for accumulating T_r
        T_r_accumulator = 0
        for r in eachrow(model.data[findall(model.data.id .== s), :])
            # loop through data rows for eachc subject
            
            # track how much time has been spent in state
            T_r_accumulator += r.tstop - r.tstart
            #T_r[r.statefrom] += r.tstop - r.tstart
            if(r.statefrom != r.stateto)
                # if a state transition happens then increment time spent in state in T_r
                # and increment transition by 1 in n_rs
                # if no transition happens then accumulated state time is ignored
                T_r[r.statefrom] += T_r_accumulator
                T_r_accumulator = 0
                n_rs[r.statefrom, r.stateto] += 1
            end
        end
    end
    return n_rs ./ T_r
end

# take matrix of crude exponential rates
# if exp then plug in
# if exp with covariates plug in and center covariates
# if weibull
# if weibull with covariates
# if Gompertz
# if Gompertz with covariates

#exp_rates = MultistateModels.calculate_crude(model)



""" 
    compute_statetable(dat, tmat)

Internal function to compute a table with observed transition counts. 
"""
function compute_statetable(dat, tmat)
    
    # initialize matrix of zeros
    transmat = zeros(Int64, size(tmat))

    # grab subject indices
    uinds = unique(dat.id)
    subjectindices = map(x -> findall(dat.id .== x), uinds)

    # outer loop over subjects
    for s in eachindex(subjectindices)
        # inner loop over data for each subject
        for r in eachindex(subjectindices[s])
            transmat[dat.statefrom[subjectindices[s][r]], 
                     dat.stateto[subjectindices[s][r]]] += 1
        end
    end

    # return the matrix of state transitions
    return transmat
end

"""
    statetable(model::MultistateModel, groups::Vararg{Symbol})

Generate a table with counts of observed transitions.

# Arguments

- model: multistate model object.
- groups: variables on which to stratify the tables of transition counts.
"""
function statetable(model::MultistateModel, groups::Vararg{Symbol})

    # apply groupby to get all different combinations of data frames
    if length(groups) == 0
        stable, groupkeys = compute_statetable(model.data, model.tmat), "Overall"
    else
        gdat = groupby(model.data, collect(groups))
        stable, groupkeys = map(x -> compute_statetable(gdat[x], model.tmat), collect(1:length(gdat))), collect(keys(gdat))
    end
    
    # return a data structure that contains the transmats and the corresponding keys
    return (stable, groupkeys)
end

"""
    initialize_parameters(model::MultistateModel, groups::Symbol...)

# Arguments
- model: MultistateModel object
- groups: different categorical variables to subset on (e.g. male, trt, etc.)
"""
function initialize_parameters(model::MultistateModel, groups::Symbol...)
    # obtain data from multistatemodel object
    data = model.hazards[1].data

    # grab the model covariates which are categorical and use that to select categorical variables from the design matrix

    # apply state table summary to achieve crude rate by dividing the two summary tables by each other (?)
    trt_a ./ trt_b

end

""" 
    crudeinit(transmat, tmat)

Return a matrix with initial intensity values. 

# Arguments
- transmat: matrix of counts of state transitions
- tmat: matrix with allowable transitions
"""
function crudeinit(transmat, tmat)
    # set transmat entry equal to zero if it's not an allowable transition
    transmat[tmat .== 0] .= 0

    # obtain the minimum positive count of transmat and divide it by 2
    half_min_count = minimum(transmat[transmat.>0])/2

    # replace transmat entries that have allowable transitions (tmat>0) and are currently zero (transmat==0) with half_min_count
    transmat

    # take row sum of new transmat and calculate the proportions for each row
    row_sum = sum(transmat, dims=2)
    q_crude_mat = transmat ./ row_sum

    # set diagonal equal to the negative of off diagonals 
    q_crude_mat[diaginds(q_crude_mat)] = -row_sum

    # return the matrix of initial intensity values
    return q_crude_mat
end
