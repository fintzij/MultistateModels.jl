module TestFixtures

using DataFrames
using StatsModels
using MultistateModels
using MultistateModels: SamplePath, MultistateModelFitted

export baseline_exact_data,
       baseline_exact_covariates,
       censoring_panel_data,
       duplicate_transition_data,
       noncontiguous_state_data,
       subject_id_df,
       subject_id_groups,
       exp_hazard,
       wei_hazard,
       gom_hazard,
       toy_expwei_model,
       toy_gompertz_model,
       toy_two_state_transition_model,
       toy_two_state_exp_model,
       toy_absorbing_start_model,
       toy_fitted_exact_model,
       make_subjdat_covariate_panel,
       make_subjdat_baseline_panel,
       make_subjdat_single_observation_panel,
       make_subjdat_exact_match_panel,
       make_subjdat_constant_covariates_panel,
       make_subjdat_single_row_full_panel,
       make_subjdat_sojourn_panel,
       # TVC fixtures
       toy_tvc_exp_model,
       toy_tvc_wei_model,
       toy_tvc_gom_model,
       toy_illness_death_tvc_model,
       toy_semi_markov_tvc_model,
       TVC_MULTI_CHANGE_CONFIG

const INTERCEPT_ONLY = @formula(0 ~ 1)

"""Two-interval panel data without covariates (baseline-only hazards)."""
function baseline_exact_data()
    return DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2]
    )
end

"""Two-interval panel data with covariates `age` and `trt`."""
function baseline_exact_covariates()
    df = baseline_exact_data()
    df.age = [50, 51]
    df.trt = [0, 1]
    return df
end

"""Panel data with a censored interval (stateto = 0) that relies on CensoringPatterns."""
function censoring_panel_data()
    return DataFrame(
        id = [1, 1],
        tstart = [0.0, 1.0],
        tstop = [1.0, 2.0],
        statefrom = [1, 1],
        stateto = [2, 0],
        obstype = [1, 3]
    )
end

"""Generic three-row panel data used by duplicate-transition tests."""
function duplicate_transition_data()
    return DataFrame(
        id = [1, 1, 1],
        tstart = [0.0, 1.0, 2.0],
        tstop = [1.0, 2.0, 3.0],
        statefrom = [1, 2, 3],
        stateto = [2, 3, 3],
        obstype = [1, 1, 3]
    )
end

"""Data with states 1, 2, and 4 to trigger non-contiguous state validation."""
function noncontiguous_state_data()
    return DataFrame(
        id = [1, 1, 1],
        tstart = [0.0, 1.0, 2.0],
        tstop = [1.0, 2.0, 3.0],
        statefrom = [1, 2, 4],
        stateto = [2, 4, 4],
        obstype = [1, 1, 3]
    )
end

"""DataFrame containing a repeated ID structure for subjinds tests."""
function subject_id_df()
    return DataFrame(id = [1, 2, 2, 3, 3, 3, 42, 42])
end

"""Expected grouping of indices for the subject_id_df fixture."""
subject_id_groups() = [[1], [2, 3], [4, 5, 6], [7, 8]]

exp_hazard(; formula = INTERCEPT_ONLY, from::Int = 1, to::Int = 2, kwargs...) =
    Hazard(formula, "exp", from, to; kwargs...)

wei_hazard(; formula = INTERCEPT_ONLY, from::Int = 1, to::Int = 2, kwargs...) =
    Hazard(formula, "wei", from, to; kwargs...)

gom_hazard(; formula = INTERCEPT_ONLY, from::Int = 1, to::Int = 2, kwargs...) =
    Hazard(formula, "gom", from, to; kwargs...)

"""Return a minimal 3-state exponential/weibull model with age/trt covariates."""
function toy_expwei_model()
    data = DataFrame(
        id = [1, 2, 3],
        tstart = zeros(3),
        tstop = fill(10.0, 3),
        statefrom = [1, 2, 1],
        stateto = [3, 3, 1],
        obstype = ones(Int, 3),
        trt = [0, 1, 0],
        age = [23, 32, 50]
    )

    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h13 = Hazard(@formula(0 ~ 1 + trt * age), "exp", 1, 3)
    h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
    h23 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 3)

    model = multistatemodel(h12, h13, h21, h23; data = data)
    return (; model, data)
end

"""Return a fitted-model wrapper around `toy_expwei_model()` to trigger exact-data shortcuts."""
function toy_fitted_exact_model()
    fixture = toy_expwei_model()
    base_model = fixture.model
    nsubj = length(base_model.subjectindices)
    stored_loglik = (loglik = -42.0, subj_lml = fill(-1.0, nsubj))

    fitted = MultistateModelFitted(
        base_model.data,
        base_model.parameters,
        stored_loglik,
        nothing,  # vcov
        nothing,  # ij_vcov
        nothing,  # jk_vcov
        nothing,  # subject_gradients
        base_model.hazards,
        base_model.totalhazards,
        base_model.tmat,
        base_model.emat,
        base_model.hazkeys,
        base_model.subjectindices,
        base_model.SubjectWeights,
        base_model.ObservationWeights,
        base_model.CensoringPatterns,
        base_model.markovsurrogate,
        nothing,
        nothing,
        base_model.modelcall,
    )

    return (; model = fitted, loglik = stored_loglik)
end

"""Return a minimal Gompertz model with and without covariate adjustment."""
function toy_gompertz_model()
    data = DataFrame(
        id = [1, 2],
        tstart = zeros(2),
        tstop = fill(10.0, 2),
        statefrom = ones(Int, 2),
        stateto = ones(Int, 2),
        obstype = ones(Int, 2),
        trt = [0, 1]
    )

    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h13 = Hazard(@formula(0 ~ 1 + trt), "gom", 1, 3)
    h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)

    model = multistatemodel(h12, h13, h23; data = data)
    return (; model, data)
end

"""Two-state transition model plus canned sample paths used in log-likelihood tests."""
function toy_two_state_transition_model()
    h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
    h21 = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1)

    data = DataFrame(
        id = [1, 1, 1, 2, 2, 2],
        tstart = [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
        tstop = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
        statefrom = [1, 1, 1, 1, 1, 1],
        stateto = [2, 2, 1, 2, 1, 2],
        obstype = [1, 2, 1, 1, 1, 2],
        trt = [0, 1, 0, 1, 0, 1]
    )

    model = multistatemodel(h12, h21; data = data)
    set_parameters!(
        model,
        (h12 = [log(0.1), log(2.0)],
         h21 = [log(1.0), log(0.1), log(2.0)])
    )

    path1 = SamplePath(1, [0.0, 8.2, 13.2, 30.0], [1, 2, 1, 1])
    path2 = SamplePath(2, [0.0, 1.7, 27.2, 28.5, 29.3], [1, 2, 1, 2, 1])

    return (; model, data, paths = (path1 = path1, path2 = path2))
end

"""Simple two-state exponential model with adjustable rate, horizon, and optional time_transform toggle on the hazard."""
function toy_two_state_exp_model(; rate = 0.2, horizon = 50.0, time_transform::Bool = false)
    data = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [horizon],
        statefrom = [1],
        stateto = [1],
        obstype = [1]
    )

    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2; time_transform = time_transform)
    model = multistatemodel(h12; data = data)
    set_parameters!(model, (h12 = [log(rate)],))
    return (; model, rate, data)
end

"""Panel data with covariate changes for make_subjdat tests."""
function make_subjdat_covariate_panel()
    return DataFrame(
        id = fill(1, 5),
        tstart = [0.0, 3.0, 7.0, 12.0, 18.0],
        tstop = [3.0, 7.0, 12.0, 18.0, 25.0],
        statefrom = fill(1, 5),
        stateto = fill(1, 5),
        obstype = fill(2, 5),
        trt = [0, 1, 1, 0, 1],
        age = [50, 50, 55, 55, 60]
    )
end

"""Return a copy of `toy_expwei_model()` where subject 1 starts in absorbing state 3."""
function toy_absorbing_start_model()
    fixture = toy_expwei_model()
    model = deepcopy(fixture.model)
    subj_inds = model.subjectindices[1]
    model.data[subj_inds, :statefrom] .= 3
    model.data[subj_inds, :stateto] .= 3
    return (; model, data = model.data)
end

"""Baseline-only panel data with no covariates."""
function make_subjdat_baseline_panel()
    return DataFrame(
        id = fill(1, 3),
        tstart = [0.0, 5.0, 15.0],
        tstop = [5.0, 15.0, 25.0],
        statefrom = fill(1, 3),
        stateto = fill(1, 3),
        obstype = fill(2, 3)
    )
end

"""Single-row panel with a covariate column."""
function make_subjdat_single_observation_panel()
    return DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [10.0],
        statefrom = [1],
        stateto = [1],
        obstype = [2],
        trt = [1]
    )
end

"""Two-row panel where covariate changes align with path times."""
function make_subjdat_exact_match_panel()
    return DataFrame(
        id = [1, 1],
        tstart = [0.0, 5.0],
        tstop = [5.0, 10.0],
        statefrom = [1, 1],
        stateto = [1, 1],
        obstype = [2, 2],
        trt = [0, 1]
    )
end

"""Panel where covariates are present but never change."""
function make_subjdat_constant_covariates_panel()
    return DataFrame(
        id = fill(1, 4),
        tstart = [0.0, 5.0, 10.0, 15.0],
        tstop = [5.0, 10.0, 15.0, 20.0],
        statefrom = fill(1, 4),
        stateto = fill(1, 4),
        obstype = fill(2, 4),
        trt = fill(1, 4)
    )
end

"""Single-row panel with multiple covariates (trt + age)."""
function make_subjdat_single_row_full_panel()
    return DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [15.0],
        statefrom = [1],
        stateto = [1],
        obstype = [2],
        trt = [0],
        age = [45]
    )
end

"""Panel used to verify sojourn time accumulation."""
function make_subjdat_sojourn_panel()
    return DataFrame(
        id = fill(1, 3),
        tstart = [0.0, 10.0, 20.0],
        tstop = [10.0, 20.0, 30.0],
        statefrom = fill(1, 3),
        stateto = fill(1, 3),
        obstype = fill(2, 3),
        trt = [0, 0, 1]
    )
end

#==============================================================================#
#                     TIME-VARYING COVARIATE (TVC) FIXTURES                    #
#==============================================================================#

"""
Standard TVC configuration with multiple covariate change points.
- t_changes: times at which covariate changes
- x_values: covariate values in each interval [0, t_changes[1]), [t_changes[1], t_changes[2]), ...
- horizon: maximum observation time
"""
const TVC_MULTI_CHANGE_CONFIG = (
    t_changes = [1.5, 3.0],
    x_values = [0.5, 1.5, 2.5],
    horizon = 5.0,
    # Hazard parameters for different families
    exp_rate = 0.35,
    wei_shape = 1.35,
    wei_scale = 0.4,
    gom_shape = 0.6,
    gom_scale = 0.4,
    beta = 0.6,  # covariate coefficient
)

"""
Build TVC data with multiple covariate change points.
Returns DataFrame with one row per interval.
"""
function _build_tvc_data(; 
    t_changes::Vector{Float64} = TVC_MULTI_CHANGE_CONFIG.t_changes,
    x_values::Vector{Float64} = Float64.(TVC_MULTI_CHANGE_CONFIG.x_values),
    horizon::Float64 = TVC_MULTI_CHANGE_CONFIG.horizon,
    n_subjects::Int = 1
)
    @assert length(x_values) == length(t_changes) + 1 "x_values should have one more element than t_changes"
    
    # Build time grid: [0, t_changes[1]], [t_changes[1], t_changes[2]], ..., [t_changes[end], horizon]
    tstart_grid = vcat(0.0, t_changes)
    tstop_grid = vcat(t_changes, horizon)
    n_intervals = length(tstart_grid)
    
    rows = []
    for subj in 1:n_subjects
        for i in 1:n_intervals
            push!(rows, (
                id = subj,
                tstart = tstart_grid[i],
                tstop = tstop_grid[i],
                statefrom = 1,
                stateto = 1,  # will be updated by simulation
                obstype = 1,
                x = x_values[i]
            ))
        end
    end
    
    return DataFrame(rows)
end

"""
Two-state exponential model with time-varying covariate (multiple change points).
Default uses PH effect; set `linpred_effect=:aft` for AFT.
"""
function toy_tvc_exp_model(; 
    linpred_effect::Symbol = :ph,
    t_changes::Vector{Float64} = TVC_MULTI_CHANGE_CONFIG.t_changes,
    x_values::Vector{Float64} = Float64.(TVC_MULTI_CHANGE_CONFIG.x_values),
    horizon::Float64 = TVC_MULTI_CHANGE_CONFIG.horizon,
    rate::Float64 = TVC_MULTI_CHANGE_CONFIG.exp_rate,
    beta::Float64 = TVC_MULTI_CHANGE_CONFIG.beta
)
    data = _build_tvc_data(; t_changes, x_values, horizon)
    
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = linpred_effect, time_transform = true)
    model = multistatemodel(h12; data = data)
    set_parameters!(model, (h12 = [log(rate), beta],))
    
    return (; model, data, config = (; t_changes, x_values, horizon, rate, beta, linpred_effect))
end

"""
Two-state Weibull model with time-varying covariate (multiple change points).
Default uses PH effect; set `linpred_effect=:aft` for AFT.
"""
function toy_tvc_wei_model(;
    linpred_effect::Symbol = :ph,
    t_changes::Vector{Float64} = TVC_MULTI_CHANGE_CONFIG.t_changes,
    x_values::Vector{Float64} = Float64.(TVC_MULTI_CHANGE_CONFIG.x_values),
    horizon::Float64 = TVC_MULTI_CHANGE_CONFIG.horizon,
    shape::Float64 = TVC_MULTI_CHANGE_CONFIG.wei_shape,
    scale::Float64 = TVC_MULTI_CHANGE_CONFIG.wei_scale,
    beta::Float64 = TVC_MULTI_CHANGE_CONFIG.beta
)
    data = _build_tvc_data(; t_changes, x_values, horizon)
    
    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = linpred_effect, time_transform = true)
    model = multistatemodel(h12; data = data)
    set_parameters!(model, (h12 = [log(shape), log(scale), beta],))
    
    return (; model, data, config = (; t_changes, x_values, horizon, shape, scale, beta, linpred_effect))
end

"""
Two-state Gompertz model with time-varying covariate (multiple change points).
Default uses PH effect; set `linpred_effect=:aft` for AFT.
"""
function toy_tvc_gom_model(;
    linpred_effect::Symbol = :ph,
    t_changes::Vector{Float64} = TVC_MULTI_CHANGE_CONFIG.t_changes,
    x_values::Vector{Float64} = Float64.(TVC_MULTI_CHANGE_CONFIG.x_values),
    horizon::Float64 = TVC_MULTI_CHANGE_CONFIG.horizon,
    shape::Float64 = TVC_MULTI_CHANGE_CONFIG.gom_shape,
    scale::Float64 = TVC_MULTI_CHANGE_CONFIG.gom_scale,
    beta::Float64 = TVC_MULTI_CHANGE_CONFIG.beta
)
    data = _build_tvc_data(; t_changes, x_values, horizon)
    
    h12 = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect = linpred_effect, time_transform = true)
    model = multistatemodel(h12; data = data)
    set_parameters!(model, (h12 = [log(shape), log(scale), beta],))
    
    return (; model, data, config = (; t_changes, x_values, horizon, shape, scale, beta, linpred_effect))
end

"""
Illness-death model (3-state) with time-varying covariate.
States: 1 (healthy) → 2 (ill) → 3 (dead), and 1 → 3 directly.
"""
function toy_illness_death_tvc_model(;
    linpred_effect::Symbol = :ph,
    t_changes::Vector{Float64} = TVC_MULTI_CHANGE_CONFIG.t_changes,
    x_values::Vector{Float64} = Float64.(TVC_MULTI_CHANGE_CONFIG.x_values),
    horizon::Float64 = TVC_MULTI_CHANGE_CONFIG.horizon
)
    # Build data with TVC
    tstart_grid = vcat(0.0, t_changes)
    tstop_grid = vcat(t_changes, horizon)
    n_intervals = length(tstart_grid)
    
    rows = []
    for i in 1:n_intervals
        push!(rows, (
            id = 1,
            tstart = tstart_grid[i],
            tstop = tstop_grid[i],
            statefrom = 1,
            stateto = 1,
            obstype = 1,
            x = x_values[i]
        ))
    end
    data = DataFrame(rows)
    
    # Define hazards for illness-death model
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = linpred_effect)
    h13 = Hazard(@formula(0 ~ x), "wei", 1, 3; linpred_effect = linpred_effect)
    h23 = Hazard(@formula(0 ~ x), "exp", 2, 3; linpred_effect = linpred_effect)
    
    model = multistatemodel(h12, h13, h23; data = data)
    set_parameters!(model, (
        h12 = [log(0.3), 0.4],   # exp: log(rate), beta
        h13 = [log(1.2), log(0.2), 0.3],  # wei: log(shape), log(scale), beta
        h23 = [log(0.5), 0.2]    # exp: log(rate), beta
    ))
    
    return (; model, data, config = (; t_changes, x_values, horizon, linpred_effect))
end

"""
Semi-Markov model with time-varying covariate.
Tests sojourn time reset interaction with TVC.
Uses Weibull hazard (shape != 1) to verify sojourn time handling.
"""
function toy_semi_markov_tvc_model(;
    linpred_effect::Symbol = :ph,
    t_changes::Vector{Float64} = [2.0, 4.0],
    x_values::Vector{Float64} = [0.5, 1.0, 1.5],
    horizon::Float64 = 6.0
)
    # Build data: subject starts in state 1 with TVC
    tstart_grid = vcat(0.0, t_changes)
    tstop_grid = vcat(t_changes, horizon)
    n_intervals = length(tstart_grid)
    
    rows = []
    for i in 1:n_intervals
        push!(rows, (
            id = 1,
            tstart = tstart_grid[i],
            tstop = tstop_grid[i],
            statefrom = 1,
            stateto = 1,
            obstype = 1,
            x = x_values[i]
        ))
    end
    data = DataFrame(rows)
    
    # Two-state reversible model with Weibull (semi-Markov) hazards
    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = linpred_effect)
    h21 = Hazard(@formula(0 ~ x), "wei", 2, 1; linpred_effect = linpred_effect)
    
    model = multistatemodel(h12, h21; data = data)
    set_parameters!(model, (
        h12 = [log(1.5), log(0.5), 0.3],  # wei: log(shape), log(scale), beta
        h21 = [log(1.3), log(0.4), 0.2]   # wei: log(shape), log(scale), beta
    ))
    
    return (; model, data, config = (; t_changes, x_values, horizon, linpred_effect))
end

end # module
