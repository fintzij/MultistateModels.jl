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


""" 
    compute_statetable(dat, tmat)

Internal function to compute a table with observed transition counts. 
"""
# function compute_statetable(dat, tmat)
    
#     # initialize matrix of zeros
#     transmat = zeros(Int64, size(tmat))

#     # grab subject indices
#     uinds = unique(dat.id)
#     subjectindices = map(x -> findall(dat.id .== x), uinds)

#     # outer loop over subjects
#     for s in eachindex(subjectindices)
#         # inner loop over data for each subject
#         for r in eachindex(subjectindices[s])
#             transmat[dat.statefrom[subjectindices[s][r]], 
#                      dat.stateto[subjectindices[s][r]]] += 1
#         end
#     end

#     # return the matrix of state transitions
#     return transmat
# end

"""
    statetable(model::MultistateProcess, groups::Vararg{Symbol})

Generate a table with counts of observed transitions.

# Arguments

- model: multistate model object.
- groups: variables on which to stratify the tables of transition counts.
"""
function statetable(model::MultistateProcess, groups::Vararg{Symbol})

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
    initialize_parameters(model::MultistateProcess, groups::Symbol...)

# Arguments
- model: MultistateModel object
- groups: different categorical variables to subset on (e.g. male, trt, etc.)
"""

function initialize_parameters(model::MultistateProcess, groups::Symbol...)
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

"""
    make_constraints(cons::Vector{Expr}, lcons::Vector{Float64}, ucons::Vector{Float64})
"""
function make_constraints(;cons::Vector{Expr}, lcons::Vector{Float64}, ucons::Vector{Float64})
    if !(length(cons) == length(lcons) == length(ucons))
        @error "cons, lcons, and ucons must all be the same length."
    end
    return (cons = cons, lcons = lcons, ucons = ucons)
end

"""
    parse_constraints(cons, hazards)

Parse user-defined constraints to generate a function constraining parameters for use in the solvers provided by Optimization.jl.
"""
function parse_constraints(cons::Vector{Expr}, hazards)

    # grab parameter names
    pars_flat = reduce(vcat, map(x -> x.parnames, hazards))

    # loop through the parameters 
    for h in eachindex(cons)
        # grab the expression
        ex = cons[h]

        # sub out parameter symbols for indices
        for p in eachindex(pars_flat)  
            ex = MacroTools.postwalk(ex -> ex == pars_flat[p] ? :(parameters[$p]) : ex, ex)
        end
        
        cons[h] = ex
    end

    # manually construct the body of the constraint function
    cons_body = Meta.parse("res .= " * string(:[$(cons...),]))
    cons_call = Expr(:call, :consfun, :(res), :(parameters), :(data))
    cons_decl = Expr(:function, cons_call, cons_body)

    # evaluate to compile the function
    eval(cons_decl)

    # return the function
    return consfun
end

