"""
    optimize_surrogate(parameters, model::MultistateModel)

Optimize parameters for a Markov surrogate by minimizing the discrepancy between the cumulative curves.
"""
function optimize_surrogate(model::MultistateModel)

    # identify transient states
    transients = findall(isa.(model.totalhazards, _TotalHazardTransient))

    # identify the transient states for each hazard
    transinds  = reduce(vcat, [i * ones(Int64, length(model.totalhazards[transients[i]].components)) for i in eachindex(transients)])

    # identify unique subjects - defined by covariate histories and observation times
    udat = unique(map(x -> view(model.data, x, Not(:id)), model.subjectindices), dims = 1)
    gdat = map(x -> view(model.data, x, Not(:id)), model.subjectindices)
    uinds = indexin(udat, gdat) # for indexing into data
    ginds = indexin(gdat, udat) # for indexing into cumulative incidence curves 

    # calculate cumulative incidence for unique subjects
    cumincs = map(y -> map(x -> cumulative_incidence(model, x, view(model.data.tstop, model.subjectindices[x]), y), uinds), transients)

    # set up optimization problem
    for s in eachindex(transients)
        # origination state
        statefrom = transients[s]

        # flatten parameters
        surpars = flatview(model.markovsurrogate.parameters[model.totalhazards[statefrom].components])

        # set up objective function
        discrepancy(sp, target) = map(x -> cumulative_incidence(model, view(model.data.tstop, model.subjectindices[x]), statefrom), uinds)

    end

end

function discrepancy(parameters, control)

    # nested view of surrogate parameters
    pars = VectorOfVectors(parameters, control.model.markovsurrogate.elem_ptr)

    # compute cumulative incidence
    cuminc_surr = map(x -> cumulative_incidence(model, view(control.model.data.tstop, controlmodel.subjectindices[x], control.statefrom), control.uinds))

end