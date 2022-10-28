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
