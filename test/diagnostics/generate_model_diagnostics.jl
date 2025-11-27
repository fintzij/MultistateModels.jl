#!/usr/bin/env julia

using CairoMakie
using DataFrames
using Distributions
using Random
using StatsBase
using StatsModels

pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using MultistateModels: Hazard, multistatemodel, set_parameters!, simulate_path, call_haz, call_cumulhaz, survprob
using MultistateModels

const OUTPUT_DIR = normpath(joinpath(@__DIR__, "assets"))
mkpath(OUTPUT_DIR)
CairoMakie.activate!(type = "png", px_per_unit = 2.0)

const COVARIATE_VALUE = 1.5
const DELTA_U = sqrt(eps())
const DELTA_T = sqrt(eps())
const SIM_SAMPLES = 40_000
const DIST_GRID_POINTS = 400

const FAMILY_CONFIG = Dict(
    "exp" => (; rate = 0.35, beta = 0.6, horizon = 5.0, hazard_start = 0.0),
    "wei" => (; shape = 1.35, scale = 0.4, beta = -0.35, horizon = 5.0, hazard_start = 0.02),
    "gom" => (; shape = 0.6, scale = 0.4, beta = 0.5, horizon = 5.0, hazard_start = 0.0),
)

struct Scenario
    family::String
    effect::Symbol
    covariate_mode::Symbol
    label::String
    slug::String
    config::NamedTuple
end

function Scenario(family::String, effect::Symbol, cov_mode::Symbol)
    config = FAMILY_CONFIG[family]
    label = string(uppercasefirst(lowercase(family)), " ", uppercase(String(effect)), " ", cov_mode == :covariate ? "with covariate" : "baseline-only")
    slug = string(family, "_", effect, "_", cov_mode)
    return Scenario(family, effect, cov_mode, label, slug, config)
end

const SCENARIOS = [Scenario(fam, eff, cov) for fam in keys(FAMILY_CONFIG) for eff in (:ph, :aft) for cov in (:baseline, :covariate)]

function scenario_subject_df(scenario::Scenario)
    horizon = scenario.config.horizon
    df = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [horizon],
        statefrom = [1],
        stateto = [2],
        obstype = [1],
    )
    if scenario.covariate_mode == :covariate
        df.x = [COVARIATE_VALUE]
    end
    return df
end

function hazard_formula(scenario::Scenario)
    scenario.covariate_mode == :covariate ? @formula(0 ~ x) : @formula(0 ~ 1)
end

function scenario_parameter_vector(scenario::Scenario)
    cfg = scenario.config
    if scenario.family == "exp"
        base = [log(cfg.rate)]
    elseif scenario.family == "wei"
        base = [log(cfg.shape), log(cfg.scale)]
    elseif scenario.family == "gom"
        base = [log(cfg.shape), log(cfg.scale)]
    else
        error("Unsupported family $(scenario.family)")
    end
    return scenario.covariate_mode == :covariate ? vcat(base, [cfg.beta]) : base
end

function build_model(scenario::Scenario)
    data = scenario_subject_df(scenario)
    hazard = Hazard(
        hazard_formula(scenario),
        scenario.family,
        1,
        2;
        linpred_effect = scenario.effect,
        time_transform = true,
    )
    model = multistatemodel(hazard; data = data)
    pars = scenario_parameter_vector(scenario)
    hazname = model.hazards[1].hazname
    set_parameters!(model, NamedTuple{(hazname,)}((pars,)))
    return model, data[1, :]
end

function covariate_value(scenario::Scenario)
    return scenario.covariate_mode == :covariate ? COVARIATE_VALUE : 0.0
end

function expected_curves(scenario::Scenario, times_h::Vector{Float64}, times_cs::Vector{Float64})
    cfg = scenario.config
    xval = covariate_value(scenario)
    beta = scenario.covariate_mode == :covariate ? cfg.beta : 0.0
    if scenario.family == "exp"
        base_rate = cfg.rate
        rate = scenario.effect == :ph ? base_rate * exp(beta * xval) : base_rate * exp(-beta * xval)
        haz_expected = fill(rate, length(times_h))
        cum_expected = rate .* times_cs
    elseif scenario.family == "wei"
        shape = cfg.shape
        scale = cfg.scale
        multiplier = scenario.effect == :ph ? exp(beta * xval) : exp(-shape * beta * xval)
        haz_expected = shape * scale .* times_h .^ (shape - 1) .* multiplier
        cum_expected = scale * multiplier .* times_cs .^ shape
    elseif scenario.family == "gom"
        shape = cfg.shape
        scale = cfg.scale
        linpred = beta * xval
        if scenario.effect == :ph
            haz_expected = scale * shape .* exp.(shape .* times_h .+ linpred)
            cum_expected = scale * exp(linpred) .* (exp.(shape .* times_cs) .- 1)
        else
            time_scale = exp(-linpred)
            haz_expected = scale * shape * time_scale .* exp.(shape * time_scale .* times_h)
            cum_expected = scale .* (exp.(shape * time_scale .* times_cs) .- 1)
        end
    else
        error("Unsupported family $(scenario.family)")
    end
    surv_expected = exp.(-cum_expected)
    return (; haz_expected, cum_expected, surv_expected)
end

function distribution_functions(scenario::Scenario)
    cfg = scenario.config
    xval = covariate_value(scenario)
    beta = scenario.covariate_mode == :covariate ? cfg.beta : 0.0
    if scenario.family == "exp"
        base_rate = cfg.rate
        rate = scenario.effect == :ph ? base_rate * exp(beta * xval) : base_rate * exp(-beta * xval)
        cumhaz = t -> rate * t
        hazard = _ -> rate
    elseif scenario.family == "wei"
        shape = cfg.shape
        scale = cfg.scale
        multiplier = scenario.effect == :ph ? exp(beta * xval) : exp(-shape * beta * xval)
        cumhaz = t -> scale * multiplier * (t^shape)
        hazard = t -> shape * scale * multiplier * (t^(shape - 1))
    elseif scenario.family == "gom"
        shape = cfg.shape
        scale = cfg.scale
        linpred = beta * xval
        if scenario.effect == :ph
            cumhaz = t -> scale * exp(linpred) * (exp(shape * t) - 1)
            hazard = t -> scale * shape * exp(shape * t + linpred)
        else
            time_scale = exp(-linpred)
            cumhaz = t -> scale * (exp(shape * time_scale * t) - 1)
            hazard = t -> scale * shape * time_scale * exp(shape * time_scale * t)
        end
    else
        error("Unsupported family $(scenario.family)")
    end
    cdf = t -> t <= 0 ? 0.0 : 1 - exp(-cumhaz(t))
    pdf = t -> t <= 0 ? 0.0 : hazard(t) * exp(-cumhaz(t))
    return cdf, pdf
end

function truncated_distribution_functions(scenario::Scenario)
    cdf_base, pdf_base = distribution_functions(scenario)
    horizon = scenario.config.horizon
    mass = cdf_base(horizon)
    mass > 0 || error("Scenario $(scenario.slug) has zero probability of transitioning before horizon=$(horizon)")

    cdf_trunc = function (t)
        if t <= 0
            return 0.0
        elseif t >= horizon
            return 1.0
        else
            return cdf_base(t) / mass
        end
    end

    pdf_trunc = function (t)
        if t <= 0 || t >= horizon
            return 0.0
        else
            return pdf_base(t) / mass
        end
    end

    return cdf_trunc, pdf_trunc
end

function hazard_time_grid(scenario::Scenario)
    horizon = scenario.config.horizon
    start = get(scenario.config, :hazard_start, 0.0)
    haz_start = start == 0.0 ? 0.0 : start
    return collect(range(haz_start, horizon; length = 200)), collect(range(0.0, horizon; length = 200))
end

function collect_event_durations(model, nsamples; time_transform::Bool, rng::AbstractRNG)
    durations = Vector{Float64}(undef, nsamples)
    collected = 0
    attempts = 0
    max_attempts = nsamples * 200
    while collected < nsamples
        path = simulate_path(model, 1, DELTA_U, DELTA_T; time_transform = time_transform, rng = rng)
        attempts += 1
        attempts > max_attempts && error("Exceeded maximum attempts without enough uncensored paths")
        if path.states[end] != path.states[1]
            collected += 1
            durations[collected] = path.times[end] - path.times[1]
        end
    end
    return durations
end

function plot_function_panel(scenario::Scenario, model, subj_row)
    times_h, times_cs = hazard_time_grid(scenario)
    curves = expected_curves(scenario, times_h, times_cs)
    hazard = model.hazards[1]
    pars = model.parameters[1]
    haz_calc = [call_haz(t, pars, subj_row, hazard; give_log = false, apply_transform = false) for t in times_h]
    haz_tt = [call_haz(t, pars, subj_row, hazard; give_log = false, apply_transform = true) for t in times_h]
    cum_calc = [call_cumulhaz(0.0, t, pars, subj_row, hazard; give_log = false, apply_transform = false) for t in times_cs]
    cum_tt = [call_cumulhaz(0.0, t, pars, subj_row, hazard; give_log = false, apply_transform = true) for t in times_cs]
    surv_calc = [survprob(0.0, t, model.parameters, subj_row, model.totalhazards[1], model.hazards; give_log = false, apply_transform = false) for t in times_cs]
    surv_tt = [survprob(0.0, t, model.parameters, subj_row, model.totalhazards[1], model.hazards; give_log = false, apply_transform = true) for t in times_cs]

    fig = Figure(size = (1200, 900))
    colors = Dict(:expected => :black, :calc => :dodgerblue, :tt => :darkorange)

    ax1 = Axis(fig[1, 1], title = "Hazard", xlabel = "Time", ylabel = "h(t)")
    lines!(ax1, times_h, curves.haz_expected, color = colors[:expected], linewidth = 3, label = "analytic")
    lines!(ax1, times_h, haz_calc, color = colors[:calc], linewidth = 2, label = "call_haz")
    lines!(ax1, times_h, haz_tt, color = colors[:tt], linewidth = 2, linestyle = :dash, label = "call_haz (time transform)")
    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[1, 2], title = "Cumulative hazard", xlabel = "Time", ylabel = "Λ(t)")
    lines!(ax2, times_cs, curves.cum_expected, color = colors[:expected], linewidth = 3)
    lines!(ax2, times_cs, cum_calc, color = colors[:calc], linewidth = 2)
    lines!(ax2, times_cs, cum_tt, color = colors[:tt], linewidth = 2, linestyle = :dash)

    ax3 = Axis(fig[2, 1:2], title = "Survival", xlabel = "Time", ylabel = "S(t)")
    lines!(ax3, times_cs, curves.surv_expected, color = colors[:expected], linewidth = 3)
    lines!(ax3, times_cs, surv_calc, color = colors[:calc], linewidth = 2)
    lines!(ax3, times_cs, surv_tt, color = colors[:tt], linewidth = 2, linestyle = :dash)

    fname = joinpath(OUTPUT_DIR, "function_panel_$(scenario.slug).png")
    save(fname, fig)
    println("saved $(basename(fname))")
end

function plot_distribution_panel(scenario::Scenario, model)
    seed = hash(scenario.slug)
    rng_tt = Random.MersenneTwister(seed)
    rng_fb = Random.MersenneTwister(seed)
    durations_tt = collect_event_durations(model, SIM_SAMPLES; time_transform = true, rng = rng_tt)
    durations_fb = collect_event_durations(model, SIM_SAMPLES; time_transform = false, rng = rng_fb)

    ecdf_tt = ecdf(durations_tt)
    ecdf_fb = ecdf(durations_fb)
    horizon = scenario.config.horizon
    ts = collect(range(0.0, horizon; length = DIST_GRID_POINTS))
    cdf_fn, pdf_fn = truncated_distribution_functions(scenario)
    expected = cdf_fn.(ts)
    empirical = ecdf_fb.(ts)
    residual = empirical .- expected
    diff_curve = ecdf_tt.(ts) .- ecdf_fb.(ts)
    max_abs_diff = maximum(abs.(diff_curve))
    ylim_span = max(max_abs_diff, 1e-6)
    xs = ts

    fig = Figure(size = (1500, 600))
    ax1 = Axis(fig[1, 1], title = "ECDF vs expected", xlabel = "Duration", ylabel = "F(t)")
    lines!(ax1, ts, expected, color = :black, linewidth = 3, label = "analytic")
    lines!(ax1, ts, empirical, color = :dodgerblue, linewidth = 2, label = "simulate_path")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[1, 2], title = "ECDF residual", xlabel = "Duration", ylabel = "Empirical − Expected")
    lines!(ax2, ts, residual, color = :crimson, linewidth = 2)
    hlines!(ax2, [0.0], color = :black, linestyle = :dash)

    ax3 = Axis(fig[1, 3], title = "Time-transform parity", xlabel = "Duration", ylabel = "ΔF(t)")
    lines!(ax3, ts, diff_curve, color = :seagreen, linewidth = 2)
    hlines!(ax3, [0.0], color = :black, linestyle = :dash)
    ylims!(ax3, (-1.1 * ylim_span, 1.1 * ylim_span))
    axislegend(ax3, [LineElement(color = :seagreen)], ["F_tt − F_fallback"], position = :rb)

    ax_hist = Axis(fig[2, 1:3], title = "Density vs histogram", xlabel = "Duration", ylabel = "Density")
    hist!(ax_hist, durations_fb; bins = 80, normalization = :pdf, color = (:steelblue, 0.5), strokewidth = 0)
    lines!(ax_hist, xs, pdf_fn.(xs), color = :black, linewidth = 3)

    fname = joinpath(OUTPUT_DIR, "simulation_panel_$(scenario.slug).png")
    save(fname, fig)
    println("saved $(basename(fname)) (max |ΔF| = $(round(max_abs_diff; digits = 3)))")
end

function generate_all()
    for scenario in sort!(copy(SCENARIOS); by = s -> s.slug)
        println("\n--- $(scenario.label) ---")
        model, subj_row = build_model(scenario)
        plot_function_panel(scenario, model, subj_row)
        plot_distribution_panel(scenario, model)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end
