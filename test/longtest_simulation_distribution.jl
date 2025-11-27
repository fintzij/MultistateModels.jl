"""
Long test suite for `simulate_path` that stress-tests:

1. Baseline exponential distributional fidelity at 1e6 draws.
2. Time-transform parity (same RNG stream ⇒ identical paths).
3. Family coverage across {exp, wei, gom} × {PH, AFT} × {covariate/no covariate}.

The diagnostics here double as the statistical backstop for the plots under
`test/diagnostics/assets`.
"""

using Base.Threads
using DataFrames
using Distributions
using HypothesisTests
using MultistateModels: Hazard, multistatemodel, set_parameters!, simulate_path
using QuadGK
using Random
using Statistics
using StatsModels
using Test

include(joinpath(@__DIR__, "fixtures", "TestFixtures.jl"))
using .TestFixtures: toy_two_state_exp_model

const RNG_SEED = 0x9b5ef02d
const SAMPLE_SIZE = 1_000_000 # ≈10-minute run; ECDF stderr ≈5e-4 at q≈0.5
const EVENT_RATE = 0.18
const PANEL_HORIZON = 200.0
const QUANTILES = (0.1, 0.5, 0.9)
const NUM_QUANTILES = length(QUANTILES)
const DELTA_U = sqrt(eps())
const DELTA_T = sqrt(eps())
const PARITY_SEED = 0x4d5e6f77

"""Concrete configuration for a long-run path simulation scenario."""
struct SimulationScenario
    name::String
    family::Symbol
    effect::Symbol
    has_covariate::Bool
    horizon::Float64
    sample_size::Int
    quantile_tol::Float64 # tolerance in probability space, i.e. |F(q̂) - q|
    mean_rel_tol::Float64
    baseline::NamedTuple
    covariate_value::Float64
    seed::UInt64
end

"""Summary statistics collected for each `SimulationScenario`."""
struct ScenarioMetrics
    empirical_mean::Float64
    target_mean::Float64
    quantile_errors::NTuple{NUM_QUANTILES,Float64}
    ks_pvalue::Float64
    parity_maxdiff::Float64
    parity_pvalue::Float64
end

"""Return E[T | lower ≤ T ≤ upper] for a continuous distribution."""
function truncated_mean(dist::ContinuousUnivariateDistribution, lower::Float64, upper::Float64)
    lower_cdf = cdf(dist, lower)
    upper_cdf = cdf(dist, upper)
    mass = upper_cdf - lower_cdf
    mass > 0 || error("Truncated distribution has zero mass in [$(lower), $(upper)].")
    integrand(x) = x * pdf(dist, x)
    integral, _ = quadgk(integrand, lower, upper; rtol = 1e-8)
    return integral / mass
end

"""Return |F(q̂) − q| for each requested quantile."""
function quantile_probability_errors(samples::AbstractVector{<:Real}, dist::ContinuousUnivariateDistribution, quantiles)
    return [abs(cdf(dist, quantile(samples, q)) - q) for q in quantiles]
end

"""Compute empirical vs. analytic metrics for a given simulation scenario."""
function scenario_metrics(scenario::SimulationScenario)
    model = build_scenario_model(scenario)
    base_dist = scenario_distribution(scenario)
    target_dist = Truncated(base_dist, 0.0, scenario.horizon)
    target_mean = truncated_mean(base_dist, 0.0, scenario.horizon)

    baseline_rng = Random.MersenneTwister(scenario.seed)
    baseline_durations = collect_event_durations(model, scenario.sample_size; time_transform = false, rng = baseline_rng)
    empirical_mean = mean(baseline_durations)
    quantile_vec = quantile_probability_errors(baseline_durations, target_dist, QUANTILES)
    ks_pvalue = pvalue(ApproximateOneSampleKSTest(baseline_durations, target_dist))
    baseline_durations = nothing

    tt_model = build_scenario_model(scenario; time_transform = true)
    tt_rng = Random.MersenneTwister(scenario.seed)
    tt_durations = collect_event_durations(tt_model, scenario.sample_size; time_transform = true, rng = tt_rng)
    fallback_rng = Random.MersenneTwister(scenario.seed)
    fallback_durations = collect_event_durations(tt_model, scenario.sample_size; time_transform = false, rng = fallback_rng)
    parity_maxdiff = maximum(abs.(tt_durations .- fallback_durations))
    parity_pvalue = pvalue(ApproximateTwoSampleKSTest(tt_durations, fallback_durations))

    quantile_errors = ntuple(i -> quantile_vec[i], NUM_QUANTILES)
    return ScenarioMetrics(empirical_mean, target_mean, quantile_errors, ks_pvalue, parity_maxdiff, parity_pvalue)
end

"""Run every `SimulationScenario` (in parallel) and collect metrics."""
function evaluate_family_scenarios()
    results = Vector{ScenarioMetrics}(undef, length(FAMILY_SCENARIOS))
    @threads for idx in eachindex(FAMILY_SCENARIOS)
        results[idx] = scenario_metrics(FAMILY_SCENARIOS[idx])
    end
    return results
end

# Each scenario below consumes 1e6 Tang-disabled draws to stress every family/effect combination.
# Quantile tolerances live in probability space and are clamped at 1e-3 (≈2× the ECDF stderr) for stability.
const FAMILY_SCENARIOS = SimulationScenario[
    SimulationScenario("exp_ph_baseline", :exp, :ph, false, 1_000.0, 1_000_000, 1e-3, 0.005, (rate = 0.22, beta = 0.5), 0.0, 0x91f2d1a1),
    SimulationScenario("exp_ph_covariate", :exp, :ph, true, 1_000.0, 1_000_000, 1e-3, 0.005, (rate = 0.22, beta = 0.35), 1.3, 0x91f2d1a2),
    SimulationScenario("exp_aft_baseline", :exp, :aft, false, 1_000.0, 1_000_000, 1e-3, 0.005, (rate = 0.3, beta = 0.4), 0.0, 0x91f2d1a3),
    SimulationScenario("exp_aft_covariate", :exp, :aft, true, 1_000.0, 1_000_000, 1e-3, 0.005, (rate = 0.3, beta = 0.45), 1.1, 0x91f2d1a4),
    SimulationScenario("wei_ph_baseline", :wei, :ph, false, 1_000.0, 1_000_000, 1e-3, 0.008, (shape = 1.4, scale = 0.02, beta = 0.3), 0.0, 0x91f2d1a5),
    SimulationScenario("wei_ph_covariate", :wei, :ph, true, 1_000.0, 1_000_000, 1e-3, 0.008, (shape = 1.4, scale = 0.02, beta = 0.25), 1.2, 0x91f2d1a6),
    SimulationScenario("wei_aft_baseline", :wei, :aft, false, 1_000.0, 1_000_000, 1e-3, 0.008, (shape = 1.2, scale = 0.03, beta = 0.35), 0.0, 0x91f2d1a7),
    SimulationScenario("wei_aft_covariate", :wei, :aft, true, 1_000.0, 1_000_000, 1e-3, 0.008, (shape = 1.2, scale = 0.03, beta = 0.3), 1.4, 0x91f2d1a8),
    SimulationScenario("gom_ph_baseline", :gom, :ph, false, 1_000.0, 1_000_000, 1e-3, 0.015, (shape = 0.12, scale = 0.45, beta = 0.2), 0.0, 0x91f2d1a9),
    SimulationScenario("gom_ph_covariate", :gom, :ph, true, 1_000.0, 1_000_000, 1e-3, 0.015, (shape = 0.12, scale = 0.45, beta = 0.25), 1.5, 0x91f2d1aa),
    SimulationScenario("gom_aft_baseline", :gom, :aft, false, 1_000.0, 1_000_000, 1e-3, 0.015, (shape = 0.1, scale = 0.5, beta = 0.22), 0.0, 0x91f2d1ab),
    SimulationScenario("gom_aft_covariate", :gom, :aft, true, 1_000.0, 1_000_000, 1e-3, 0.015, (shape = 0.1, scale = 0.5, beta = 0.28), 1.25, 0x91f2d1ac),
]

"""Minimal Gompertz distribution implementation used for targets."""
struct GompertzReference <: ContinuousUnivariateDistribution
    scale::Float64
    shape::Float64
end

Distributions.minimum(::GompertzReference) = 0.0
Distributions.maximum(::GompertzReference) = Inf
Distributions.insupport(::GompertzReference, x::Real) = x >= 0

@inline function _gompertz_exp_term(dist::GompertzReference, x::Float64)
    return exp(dist.shape * x)
end

function Distributions.cdf(dist::GompertzReference, x::Real)
    x < 0 && return 0.0
    xval = float(x)
    if abs(dist.shape) < 1e-10
        return 1 - exp(-dist.scale * xval)
    else
        exp_term = _gompertz_exp_term(dist, xval)
        return 1 - exp(-dist.scale * (exp_term - 1))
    end
end

function Distributions.pdf(dist::GompertzReference, x::Real)
    x < 0 && return 0.0
    xval = float(x)
    if abs(dist.shape) < 1e-10
        return dist.scale * exp(-dist.scale * xval)
    else
        exp_term = _gompertz_exp_term(dist, xval)
        return dist.scale * dist.shape * exp_term * exp(-dist.scale * (exp_term - 1))
    end
end

function Distributions.quantile(dist::GompertzReference, p::Real)
    0.0 <= p <= 1.0 || throw(ArgumentError("Quantile probability must be between 0 and 1."))
    iszero(p) && return 0.0
    p == 1.0 && return Inf
    if abs(dist.shape) < 1e-10
        return -log1p(-p) / dist.scale
    else
        return log1p(-log1p(-p) / dist.scale) / dist.shape
    end
end

function Distributions.mean(dist::GompertzReference)
    if abs(dist.shape) < 1e-10
        return 1 / dist.scale
    end
    integrand(t) = exp(-dist.scale * (exp(dist.shape * t) - 1))
    val, _ = QuadGK.quadgk(integrand, 0.0, Inf; rtol = 1e-8)
    return val
end

"""Assemble the parameter vector expected by `set_parameters!`."""
function scenario_parameters(s::SimulationScenario)
    params = Float64[]
    if s.family == :exp
        push!(params, log(s.baseline.rate))
    elseif s.family in (:wei, :gom)
        push!(params, log(s.baseline.shape))
        push!(params, log(s.baseline.scale))
    else
        error("Unsupported family $(s.family)")
    end
    if s.has_covariate
        push!(params, s.baseline.beta)
    end
    return params
end

"""Return the analytic distribution implied by the scenario metadata."""
function scenario_distribution(s::SimulationScenario)
    linpred = s.has_covariate ? s.baseline.beta * s.covariate_value : 0.0

    if s.family == :exp
        base_rate = s.baseline.rate
        rate = s.effect == :ph ? base_rate * exp(linpred) : base_rate * exp(-linpred)
        return Exponential(1 / rate)
    elseif s.family == :wei
        shape = s.baseline.shape
        scale_param = s.baseline.scale
        λ = s.effect == :ph ? scale_param * exp(linpred) : scale_param * exp(-shape * linpred)
        θ = λ^(-1 / shape)
        return Weibull(shape, θ)
    elseif s.family == :gom
        shape = s.baseline.shape
        scale_param = s.baseline.scale
        if s.effect == :ph
            scale_eff = scale_param * exp(linpred)
            shape_eff = shape
        else
            time_scale = exp(-linpred)
            shape_eff = shape * time_scale
            if abs(shape_eff) < 1e-10
                scale_eff = scale_param * time_scale
                shape_eff = 0.0
            else
                scale_eff = scale_param
            end
        end
        return GompertzReference(scale_eff, shape_eff)
    else
        error("Unsupported family $(s.family)")
    end
end

"""Create a two-state model tailored to `s` plus optional time transform."""
function build_scenario_model(s::SimulationScenario; time_transform::Bool = false)
    data = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [s.horizon],
        statefrom = [1],
        stateto = [1],
        obstype = [1],
    )

    formula = s.has_covariate ? @formula(0 ~ 1 + x) : @formula(0 ~ 1)
    if s.has_covariate
        data.x = [s.covariate_value]
    end

    hazard = Hazard(formula, String(s.family), 1, 2; linpred_effect = s.effect, time_transform = time_transform)
    model = multistatemodel(hazard; data = data)
    set_parameters!(model, (h12 = scenario_parameters(s),))
    return model
end

"""Collect uncensored event durations by repeatedly sampling paths."""
function collect_event_durations(model, nsamples; time_transform::Bool = false, rng::AbstractRNG = Random.default_rng())
    durations = Vector{Float64}(undef, nsamples)
    collected = 0
    attempts = 0
    max_attempts = nsamples * 5

    while collected < nsamples
        attempts += 1
        attempts <= max_attempts || error("Insufficient uncensored sample paths; consider increasing PANEL_HORIZON.")

        path = simulate_path(model, 1, DELTA_U, DELTA_T; time_transform = time_transform, rng = rng)
        if path.states[end] != path.states[1]
            collected += 1
            durations[collected] = path.times[end] - path.times[1]
        end
    end

    return durations
end

# ------------------------------------------------------------------------------
# Baseline exponential reference (shared with documentation panels)
# ------------------------------------------------------------------------------

fixture = toy_two_state_exp_model(rate = EVENT_RATE, horizon = PANEL_HORIZON, time_transform = true)
base_rng = Random.MersenneTwister(RNG_SEED)
durations = collect_event_durations(fixture.model, SAMPLE_SIZE; time_transform = false, rng = base_rng)
base_dist = Exponential(1 / EVENT_RATE)
target_dist = Truncated(base_dist, 0.0, PANEL_HORIZON)

empirical_mean = mean(durations)
expected_mean = truncated_mean(base_dist, 0.0, PANEL_HORIZON)
empirical_quantiles = [quantile(durations, q) for q in QUANTILES]
target_quantiles = [quantile(target_dist, q) for q in QUANTILES]
ks_test = ApproximateOneSampleKSTest(durations, target_dist)

@testset "simulate_path long-run distribution" begin
    @test isapprox(empirical_mean, expected_mean; rtol = 0.005)

    @test maximum(abs.(empirical_quantiles .- target_quantiles)) < 0.06

    @test pvalue(ks_test) > 0.05
end

@testset "simulate_path time_transform parity (baseline)" begin
    tt_rng = Random.MersenneTwister(PARITY_SEED)
    tt_sample = collect_event_durations(fixture.model, SAMPLE_SIZE; time_transform = true, rng = tt_rng)
    fallback_rng = Random.MersenneTwister(PARITY_SEED)
    fallback_sample = collect_event_durations(fixture.model, SAMPLE_SIZE; time_transform = false, rng = fallback_rng)

    @test all(isapprox.(tt_sample, fallback_sample; atol = 1e-12, rtol = 0.0))

    ks = ApproximateTwoSampleKSTest(tt_sample, fallback_sample)
    @test pvalue(ks) > 0.99
end

@testset "simulate_path family coverage" begin
    metrics = evaluate_family_scenarios()
    for (scenario, result) in zip(FAMILY_SCENARIOS, metrics)
        mean_rel_error = abs(result.empirical_mean - result.target_mean) / result.target_mean
        @test mean_rel_error < scenario.mean_rel_tol
        @test maximum(result.quantile_errors) < scenario.quantile_tol
        @test result.ks_pvalue > 0.05
        @test result.parity_maxdiff <= 1e-8
        @test result.parity_pvalue > 0.99
    end
end

println("simulate_path long test passed: mean=$(round(empirical_mean; digits=3)), KS p-value=$(round(pvalue(ks_test); digits=3))")
println("simulate_path family coverage validated for $(length(FAMILY_SCENARIOS)) scenarios")
