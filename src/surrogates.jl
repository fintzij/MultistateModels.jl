

"""
make_surrogate_model(model::MultistateSemiMarkovModel)

Create a Markov surrogate model.

# Arguments

- model: multistate model object
"""
function make_surrogate_model(model::Union{MultistateModel, MultistateMarkovModel, MultistateSemiMarkovModel})
    MultistateModels.MultistateMarkovModel(
        model.data,
        model.markovsurrogate.parameters,
        model.markovsurrogate.hazards,
        model.totalhazards,
        model.tmat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        model.modelcall)
end

"""
make_surrogate_model(model::MultistateSemiMarkovModelCensored)

Create a Markov surrogate model with censored states.

# Arguments

- model: multistate model object
"""
function make_surrogate_model(model::Union{MultistateMarkovModelCensored,MultistateSemiMarkovModelCensored})
    MultistateModels.MultistateMarkovModelCensored(
        model.data,
        model.markovsurrogate.parameters,
        model.markovsurrogate.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        model.modelcall)
end


"""
fit_surrogate(model::MultistateSemiMarkovModelCensored)

Fit a Markov surrogate model.

# Arguments

- model: multistate model object
"""
function fit_surrogate(model; surrogate_parameters = nothing)
    surrogate = make_surrogate_model(model)
    if isnothing(surrogate_parameters)
        set_crude_init!(surrogate)
    else
        set_parameters!(surrogate, surrogate_parameters)
    end
    surrogate = fit(surrogate)
    return surrogate
end




# """
#     optimize_surrogate(parameters, model::MultistateProcess)

# Optimize parameters for a Markov surrogate by minimizing the discrepancy between the cumulative curves.
# """
# function optimize_surrogate(model::MultistateProcess)

#     # identify transient states
#     transients = findall(isa.(model.totalhazards, _TotalHazardTransient))

#     # identify the transient states for each hazard
#     transinds  = reduce(vcat, [i * ones(Int64, length(model.totalhazards[transients[i]].components)) for i in eachindex(transients)])

#     # identify unique subjects - defined by covariate histories and observation times
#     gdat = map(x -> view(model.data, x, Not(:id)), model.subjectindices)
#     udat = unique(gdat, dims = 1)
#     uinds = indexin(udat, gdat) # for indexing into data
#     ginds = indexin(gdat, udat) # for indexing into cumulative incidence curves 

#     # calculate cumulative incidence for unique subjects
#     cumincs = mapreduce(x -> cumulative_incidence(model, x, view(model.data.tstop, model.subjectindices[x])), vcat, uinds)

#     # set up optimization problem
#     for s in eachindex(transients)
#         # origination state
#         statefrom = transients[s]

#         # flatten parameters
#         surpars = flatview(model.markovsurrogate.parameters[model.totalhazards[statefrom].components])

#         # # set up objective function
#         control = SurrogateControl(model, statefrom, cumincs[:,model.totalhazards[statefrom].components], uinds, ginds)
#         # discrepancy(sp, target) = map(x -> cumulative_incidence(model, view(model.data.tstop, model.subjectindices[x]), statefrom), uinds)

#     end

# end


# # e.g., parameters = flattened vector of params for h12, h13
# # but pars = parameters for h12, h13, h21, h23
# function discrepancy(parameters, control)
#     # parameters is only the parameters corresponding to the statefrom hazards (e.g. h_12 and h_13)

#     # get statefrom and hazards
#     statefrom = control.statefrom
#     hazfrom = control.model.totalhazards[statefrom].components

#     # get surrogate parameters
#     pars = control.model.markovsurrogate.parameters

#     # indices for indexing into pars
#     parinds = mapreduce(x -> pars.elem_ptr[x]:(pars.elem_ptr[x+1]-1), vcat, hazfrom)

#     # copy parameters
#     pars.data[parinds] = parameters

#     # compute cumulative incidence
#     sum((log1p.(mapreduce(x -> _cumulative_incidence(control.model, pars, control.model.markovsurrogate.hazards, x, view(control.model.data.tstop, control.model.subjectindices[x]), statefrom), vcat, control.uinds)) - log1p.(control.targets)).^2)
# end