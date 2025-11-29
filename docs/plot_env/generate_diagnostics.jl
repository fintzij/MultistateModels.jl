#!/usr/bin/env julia

using CairoMakie
using DataFrames
using Distributions
using Random
using StatsBase
using StatsModels

pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using MultistateModels

include(normpath(joinpath(@__DIR__, "..", "..", "test", "fixtures", "TestFixtures.jl")))
using .TestFixtures: toy_two_state_exp_model

const OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "assets", "diagnostics"))
mkpath(OUTPUT_DIR)
CairoMakie.activate!(type = "png", px_per_unit = 2.0)

function single_subject_df(; horizon = 5.0, x = 1.5)
    DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [horizon],
        statefrom = [1],
        stateto = [2],
        obstype = [1],
        x = [x],
    )
end

function exp_family_curves()
    data = single_subject_df()
    hazard = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = :ph, time_transform = true)
    model = multistatemodel(hazard; data = data)
    λ0 = 0.35
    β = 0.6
    MultistateModels.set_parameters!(model, (h12 = [log(λ0), β],))
    subj = data[1, :]
    rate = λ0 * exp(β * subj.x)
    times_h = collect(range(0.0, 5.0, length = 200))
    times_cs = collect(range(0.0, 5.0, length = 200))

    haz_expected = fill(rate, length(times_h))
    haz_calc = [MultistateModels.call_haz(t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = false) for t in times_h]
    haz_tt = [MultistateModels.call_haz(t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = true) for t in times_h]

    cum_expected = rate .* times_cs
    cum_calc = [MultistateModels.call_cumulhaz(0.0, t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = false) for t in times_cs]
    cum_tt = [MultistateModels.call_cumulhaz(0.0, t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = true) for t in times_cs]

    surv_expected = exp.(-cum_expected)
    surv_calc = [MultistateModels.survprob(0.0, t, model.parameters, subj, model.totalhazards[1], model.hazards; give_log = false, apply_transform = false) for t in times_cs]
    surv_tt = [MultistateModels.survprob(0.0, t, model.parameters, subj, model.totalhazards[1], model.hazards; give_log = false, apply_transform = true) for t in times_cs]

    return (; times_h, haz_expected, haz_calc, haz_tt, times_cs, cum_expected, cum_calc, cum_tt, surv_expected, surv_calc, surv_tt)
end

function weibull_family_curves()
    data = single_subject_df()
    hazard = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect = :ph, time_transform = true)
    model = multistatemodel(hazard; data = data)
    log_shape = log(1.35)
    log_scale = log(0.4)
    β = -0.35
    MultistateModels.set_parameters!(model, (h12 = [log_shape, log_scale, β],))
    subj = data[1, :]
    shape = exp(log_shape)
    scale = exp(log_scale)
    linpred = β * subj.x
    times = collect(range(0.05, 5.0, length = 200))
    times_cs = vcat(0.0, times)

    haz_expected = shape * scale .* times .^ (shape - 1) .* exp(linpred)
    haz_calc = [MultistateModels.call_haz(t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = false) for t in times]
    haz_tt = [MultistateModels.call_haz(t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = true) for t in times]

    cum_expected = scale * exp(linpred) .* times_cs .^ shape
    cum_calc = [MultistateModels.call_cumulhaz(0.0, t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = false) for t in times_cs]
    cum_tt = [MultistateModels.call_cumulhaz(0.0, t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = true) for t in times_cs]

    surv_expected = exp.(-cum_expected)
    surv_calc = [MultistateModels.survprob(0.0, t, model.parameters, subj, model.totalhazards[1], model.hazards; give_log = false, apply_transform = false) for t in times_cs]
    surv_tt = [MultistateModels.survprob(0.0, t, model.parameters, subj, model.totalhazards[1], model.hazards; give_log = false, apply_transform = true) for t in times_cs]

    return (; times_h = times, haz_expected, haz_calc, haz_tt, times_cs, cum_expected, cum_calc, cum_tt, surv_expected, surv_calc, surv_tt)
end

function gompertz_family_curves()
    data = single_subject_df()
    hazard = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect = :ph, time_transform = true)
    model = multistatemodel(hazard; data = data)
    log_shape = log(0.6)
    log_scale = log(0.4)
    β = 0.5
    MultistateModels.set_parameters!(model, (h12 = [log_shape, log_scale, β],))
    subj = data[1, :]
    shape = exp(log_shape)
    scale = exp(log_scale)
    linpred = β * subj.x
    times = collect(range(0.0, 5.0, length = 200))

    haz_expected = scale * shape .* exp.(shape .* times .+ linpred)
    haz_calc = [MultistateModels.call_haz(t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = false) for t in times]
    haz_tt = [MultistateModels.call_haz(t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = true) for t in times]

    function gompertz_cum(t)
        if abs(shape) < 1e-10
            return scale * exp(linpred) * t
        else
            return scale * exp(linpred) * (exp(shape * t) - 1)
        end
    end

    times_cs = times
    cum_expected = gompertz_cum.(times_cs)
    cum_calc = [MultistateModels.call_cumulhaz(0.0, t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = false) for t in times_cs]
    cum_tt = [MultistateModels.call_cumulhaz(0.0, t, model.parameters[1], subj, model.hazards[1]; give_log = false, apply_transform = true) for t in times_cs]

    surv_expected = exp.(-cum_expected)
    surv_calc = [MultistateModels.survprob(0.0, t, model.parameters, subj, model.totalhazards[1], model.hazards; give_log = false, apply_transform = false) for t in times_cs]
    surv_tt = [MultistateModels.survprob(0.0, t, model.parameters, subj, model.totalhazards[1], model.hazards; give_log = false, apply_transform = true) for t in times_cs]

    return (; times_h = times, haz_expected, haz_calc, haz_tt, times_cs, cum_expected, cum_calc, cum_tt, surv_expected, surv_calc, surv_tt)
end

function plot_family_panel(name::String, curves)
    fig = Figure(resolution = (1200, 900))
    colors = Dict(
        :expected => :black,
        :calc => :dodgerblue,
        :tt => :darkorange,
    )

    ax1 = Axis(fig[1, 1], title = "Hazard", xlabel = "Time", ylabel = "h(t)")
    lines!(ax1, curves.times_h, curves.haz_expected, color = colors[:expected], linewidth = 3, label = "Expected")
    lines!(ax1, curves.times_h, curves.haz_calc, color = colors[:calc], linewidth = 2, label = "call_haz (no time transformation)")
    lines!(ax1, curves.times_h, curves.haz_tt, color = colors[:tt], linewidth = 2, linestyle = :dash, label = "call_haz (time transformation)")
    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[1, 2], title = "Cumulative hazard", xlabel = "Time", ylabel = "Λ(t)")
    lines!(ax2, curves.times_cs, curves.cum_expected, color = colors[:expected], linewidth = 3)
    lines!(ax2, curves.times_cs, curves.cum_calc, color = colors[:calc], linewidth = 2)
    lines!(ax2, curves.times_cs, curves.cum_tt, color = colors[:tt], linewidth = 2, linestyle = :dash)

    ax3 = Axis(fig[2, 1:2], title = "Survival", xlabel = "Time", ylabel = "S(t)")
    lines!(ax3, curves.times_cs, curves.surv_expected, color = colors[:expected], linewidth = 3)
    lines!(ax3, curves.times_cs, curves.surv_calc, color = colors[:calc], linewidth = 2)
    lines!(ax3, curves.times_cs, curves.surv_tt, color = colors[:tt], linewidth = 2, linestyle = :dash)

    save(joinpath(OUTPUT_DIR, name), fig)
    println("saved $(name)")
end

function collect_event_durations(model, nsamples; time_transform::Bool = false, rng::AbstractRNG = Random.default_rng())
    durations = Vector{Float64}(undef, nsamples)
    collected = 0
    delta_u = sqrt(eps())
    delta_t = sqrt(eps())
    subj = 1
    while collected < nsamples
        path = MultistateModels.simulate_path(model, subj, delta_u, delta_t; rng = rng, time_transform = time_transform)
        if path.states[end] != path.states[1]
            collected += 1
            durations[collected] = path.times[end] - path.times[1]
        end
    end
    return durations
end

function longtest_distributions()
    EVENT_RATE = 0.18
    HORIZON = 200.0
    samples = 200_000
    fixture = toy_two_state_exp_model(rate = EVENT_RATE, horizon = HORIZON, time_transform = true)
    rng = Random.MersenneTwister(0x9b5ef02d)
    durations = collect_event_durations(fixture.model, samples; time_transform = false, rng = rng)
    target = Truncated(Exponential(1 / EVENT_RATE), 0.0, HORIZON)

    ts = collect(range(0.0, 40.0, length = 400))
    ecdf_fn = ecdf(durations)
    empirical = ecdf_fn.(ts)
    expected = cdf.(target, ts)
    residual = empirical .- expected
    xs = collect(range(0.0, 40.0, length = 400))

    fig = Figure(resolution = (1500, 600))
    ax1 = Axis(fig[1, 1], title = "ECDF vs expected", xlabel = "Duration", ylabel = "F(t)")
    lines!(ax1, ts, expected, color = :black, linewidth = 3, label = "Truncated Exp")
    lines!(ax1, ts, empirical, color = :dodgerblue, linewidth = 2, label = "Simulation (200k draws)")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[1, 2], title = "ECDF residual", xlabel = "Duration", ylabel = "Empirical - Expected")
    lines!(ax2, ts, residual, color = :crimson, linewidth = 2)
    hlines!(ax2, [0.0], color = :black, linestyle = :dash)

    ax3 = Axis(fig[1, 3], title = "Density vs histogram", xlabel = "Duration", ylabel = "Density")
    hist!(ax3, durations; bins = 80, normalization = :pdf, color = (:steelblue, 0.5), strokewidth = 0)
    lines!(ax3, xs, pdf.(target, xs), color = :black, linewidth = 3)

    save(joinpath(OUTPUT_DIR, "longtest_distribution_panel.png"), fig)
    println("saved longtest_distribution_panel.png")

    return durations, fixture.model
end

function time_transform_parity(model)
    samples = 200_000
    tt_rng = Random.MersenneTwister(0x4d5e6f77)
    fallback_rng = Random.MersenneTwister(0x4d5e6f77)
    tt = collect_event_durations(model, samples; time_transform = true, rng = tt_rng)
    fallback = collect_event_durations(model, samples; time_transform = false, rng = fallback_rng)
    ts = collect(range(0.0, 40.0, length = 400))
    ecdf_tt = ecdf(tt).(ts)
    ecdf_fb = ecdf(fallback).(ts)
    diff_curve = ecdf_tt .- ecdf_fb
    max_abs_diff = maximum(abs.(diff_curve))
    ylim_span = max(max_abs_diff, 1e-6)

    fig = Figure(resolution = (1200, 600))
    ax1 = Axis(fig[1, 1], title = "ECDF comparison", xlabel = "Duration", ylabel = "F(t)")
    lines!(ax1, ts, ecdf_tt, color = :darkorange, linewidth = 3, label = "time transformation")
    lines!(ax1, ts, ecdf_fb, color = :dodgerblue, linewidth = 2, linestyle = :dash, label = "fallback")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[1, 2], title = "ECDF difference (TT − fallback)", xlabel = "Duration", ylabel = "ΔF(t)")
    lines!(ax2, ts, diff_curve, color = :crimson, linewidth = 2)
    hlines!(ax2, [0.0], color = :black, linestyle = :dash)
    ylims!(ax2, (-1.1 * ylim_span, 1.1 * ylim_span))

    save(joinpath(OUTPUT_DIR, "time_transform_parity_panel.png"), fig)
    println("saved time_transform_parity_panel.png (max |ΔF| = $(max_abs_diff))")
end

function main()
    plot_family_panel("exp_family_panel.png", exp_family_curves())
    plot_family_panel("weibull_family_panel.png", weibull_family_curves())
    plot_family_panel("gompertz_family_panel.png", gompertz_family_curves())
    durations, model = longtest_distributions()
    time_transform_parity(model)
end

main()
