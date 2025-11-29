

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
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
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
        model.SubjectWeights,
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
function fit_surrogate(model; surrogate_parameters = nothing, surrogate_constraints = nothing, crude_inits = true, verbose = true)

    # initialize the surrogate
    surrogate_model = make_surrogate_model(model)

    # set parameters to supplied or crude inits
    if !isnothing(surrogate_parameters) 
        set_parameters!(surrogate_model, surrogate_parameters)
    elseif crude_inits
        set_crude_init!(surrogate_model)
    end

    # generate the constraint function and test at initial values
    if !isnothing(surrogate_constraints)
        # create the function
        consfun_surrogate = parse_constraints(surrogate_constraints.cons, surrogate_model.hazards; consfun_name = :consfun_surrogate)

        # test the initial values
        # Phase 3: Use ParameterHandling.jl flat parameters for constraint check
        initcons = consfun_surrogate(zeros(length(surrogate_constraints.cons)), get_parameters_flat(surrogate_model), nothing)
        
        badcons = findall(initcons .< surrogate_constraints.lcons .|| initcons .> surrogate_constraints.ucons)

        if length(badcons) > 0
            @error "Constraints $badcons are violated at the initial parameter values for the Markov surrogate. Consider manually setting surrogate parameters."
        end
    end

    # optimize the Markov surrogate
    if verbose
        println("Obtaining the MLE for the Markov surrogate model ...\n")
    end
    
    surrogate_fitted = fit(surrogate_model; constraints = surrogate_constraints, compute_vcov = false)

    return surrogate_fitted
end
