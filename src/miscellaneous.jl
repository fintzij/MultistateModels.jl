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
    qcrudeinit(transmat, tmat)

Return a matrix with initial intensity values. 

# Arguments
- transmat: matrix of counts of state transitions
- tmat: matrix with allowable transitions
"""
function crudeinit(transmat, tmat)
    # set transmat entry equal to zero if it's not an allowable transition
    transmat[tmat .== 0] .= 0

    # take row sum of new transmat and calculate the proportions for each row
    row_sum = sum(transmat, dims=2)
    q_crude_mat = transmat ./ row_sum

    # set diagonal equal to the negative of off diagonals 
    q_crude_mat[diaginds(q_crude_mat)] = -row_sum

    # return the matrix of initial intensity values
    return q_crude_mat
end

