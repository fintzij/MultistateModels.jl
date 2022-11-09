""" 
    statetable(model::MultistateModel)

Return a table with observed transition counts. 
"""
function statetable(model::MultistateModel)
    
    # initialize matrix of zeros
    transmat = zeros(Int64, size(model.tmat))

    # outer loop over subjects
    for s in eachindex(model.subjectindices)
        # inner loop over data for each subject
        for r in eachindex(model.subjectindices[s])
            transmat[model.data.statefrom[model.subjectindices[s][r]], 
                     model.data.stateto[model.subjectindices[s][r]]] += 1
        end
    end

    # return the matrix of state transitions
    return transmat
end

""" 
    statetable(model::MultistateModel, val, field)

Return a table with observed transition counts based on filtered fields.

# Arguments
- transmat: matrix of counts of state transitions
- tmat: matrix with allowable transitions
- val: value of the field you want to subset (e.g. male == "m" where male is the field and val is "m")
- field: variables to subset on 
"""
function statetable(model::MultistateModel, val, field)
    # obtain data from multistatemodel object
    data = model.data

    # filter multistatemodel dataframe value according to the desired field and value
    subset = data[data[:,Symbol(field)] .== val, :]
    model.data = subset
    
    # apply the statetable summary on the subsetted data frame
    transmat_subset = MultistateModel.statetable(model)
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



