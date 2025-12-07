using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SamplePath, @formula

const RNG_SEED = 0xABCD1234
const N_SUBJECTS = 1000
const N_SIM_TRAJ = 10000
const MAX_TIME = 15.0
const MCEM_TOL = 0.05
const MAX_ITER = 30
const PARAM_TOL_REL = 0.35

function compute_state_prevalence(paths::Vector{SamplePath}, eval_times::Vector{Float64}, n_states::Int)
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    n_paths = length(paths)
    for path in paths
        for (t_idx, t) in enumerate(eval_times)
            state_idx = searchsortedlast(path.times, t)
            if state_idx >= 1
                state = path.states[state_idx]
                prevalence[t_idx, state] += 1.0
            end
        end
    end
    prevalence ./= n_paths
    return prevalence
end

function generate_panel_data_illness_death(hazards, true_params;
    n_subj::Int = N_SUBJECTS,
    obs_times::Vector{Float64} = [0.0, 3.0, 6.0, 9.0, 12.0, MAX_TIME])
    nobs = length(obs_times) - 1
    template = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat(obs_times[1:end-1], n_subj),
        tstop = repeat(obs_times[2:end], n_subj),
        statefrom = ones(Int, n_subj * nobs),
        stateto = ones(Int, n_subj * nobs),
        obstype = fill(2, n_subj * nobs),
    )
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    return sim_result[1, 1]
end

function check_distributional_fidelity_mcem(hazards, true_params, fitted_params_flat;
    n_traj::Int = N_SIM_TRAJ,
    max_time::Float64 = MAX_TIME,
    eval_times::Vector{Float64} = collect(1.0:1.0:max_time),
    max_prev_diff::Float64 = 0.12,
    n_states::Int = 3)
    template = DataFrame(
        id = 1:n_traj,
        tstart = zeros(n_traj),
        tstop = fill(max_time, n_traj),
        statefrom = ones(Int, n_traj),
        stateto = ones(Int, n_traj),
        obstype = ones(Int, n_traj),
    )
    model_true = multistatemodel(hazards...; data=template)
    set_parameters!(model_true, true_params)
    model_fitted = multistatemodel(hazards...; data=template)
    idx = 1
    for (h_idx, haz) in enumerate(model_fitted.hazards)
        npar = haz.npar_total
        set_parameters!(model_fitted, h_idx, fitted_params_flat[idx:idx+npar-1])
        idx += npar
    end
    Random.seed!(RNG_SEED + 2000)
    trajectories_true = simulate(model_true; paths=true, data=false, nsim=1)
    paths_true = trajectories_true[1]
    Random.seed!(RNG_SEED + 2000)
    trajectories_fitted = simulate(model_fitted; paths=true, data=false, nsim=1)
    paths_fitted = trajectories_fitted[1]
    prev_true = compute_state_prevalence(paths_true, eval_times, n_states)
    prev_fitted = compute_state_prevalence(paths_fitted, eval_times, n_states)
    max_diff = maximum(abs.(prev_true .- prev_fitted))
    return max_diff < max_prev_diff
end

@testset "MCEM Weibull - No Covariates (standalone)" begin
    Random.seed!(RNG_SEED + 10)

    true_shape_12, true_scale_12 = 1.3, 0.15
    true_shape_23, true_scale_23 = 1.1, 0.12
    true_shape_13, true_scale_13 = 1.2, 0.06

    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h23 = [log(true_shape_23), log(true_scale_23)],
        h13 = [log(true_shape_13), log(true_scale_13)],
    )

    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)

    panel_data = generate_panel_data_illness_death((h12, h23, h13), true_params)

    model_fit = multistatemodel(h12, h23, h13; data=panel_data, surrogate=:markov)
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        compute_vcov=true,
        compute_ij_vcov=false,
        return_convergence_records=true)

    @testset "Parameter recovery" begin
        p = get_parameters(fitted; scale=:natural)
        @test isapprox(p.h12[1], true_shape_12; rtol=PARAM_TOL_REL)
        @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
        @test isapprox(p.h23[1], true_shape_23; rtol=PARAM_TOL_REL)
        @test isapprox(p.h23[2], true_scale_23; rtol=PARAM_TOL_REL)
    end

    @testset "Distributional fidelity" begin
        @test check_distributional_fidelity_mcem((h12, h23, h13), true_params, get_parameters_flat(fitted))
    end
end
