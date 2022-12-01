""" 
    statetable(model::MultistateModel)

Return a table with observed transition counts. 
"""
function statetable(dat, tmat)
    
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

function statetable(dat, tmat, fields::Symbol...)
    # apply groupby to get all different combinations of data frames
    gdat = groupby(dat, fields)

    # gather all the sub-dataframes into a vector
    subdataframes = map(_ -> DataFrame(), 1:gdat.ngroups)

    for i in 1:ngroups
        subdataframes[i] = gdat[i]
    end

    # apply statetable function to each sub-dataframe
    transmats = map(x->statetable(x, tmat), subdataframes)

    # grab labels of each grouped data frame
    keys = eachindex(gdat)

    # return a data structure that contains the transmats and the corresponding keys
end

""" 
    statetable(model::MultistateModel, fields::Symbol...)

Return a table with observed transition counts based on filtered fields.

# Arguments
- model:MultistateModel object
- field: different categorical variables to subset on (e.g. male, trt, etc.)
"""
function statetable(model::MultistateModel, fields::Symbol...)
    # obtain data from multistatemodel object
    data = model.data
    varargs = fields

    # create comprephension that obtains the unique values for each categorical variable
    unique_vals = [unique(data[!, varargs[i]]) for i in 1:length(varargs)]

    # initialize an empty dictionary (with column names) to hold all possible filtered data frames
    data_subsets = Dict(k => DataFrame([name => [] for name in names(data)]) for k in 1:sum(length, unique_vals))

    # nested loop to store the filtered data frames
    for i in 1:length(unique_vals) # for each field 
        for j in 1:length(unique_vals[i]) # for each unique value within that field
            subset = data[data[:,varargs[i]] .== unique_vals[i][j],:] # filter the data for a specific unique value of that specific field 
            print(subset) # for each iteration, how do I assign the subset to the empty dictionary above (data_subsets)?
        end
    end

    # create new msm models for each one of the filtered data frames (how do I grab the h12 and h21 from the original msm object)

    # apply state table summary counts to each of the filtered data frames 

    # return dictionary with all state table summary counts
    return(transmat)
end

"""
    initialize_parameters(model::MultistateModel, fields::Symbol...)

# Arguments
- model: MultistateModel object
- fields: different categorical variables to subset on (e.g. male, trt, etc.)
"""

function initialize_parameters(model::MultistateModel, fields::Symbol...)
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



