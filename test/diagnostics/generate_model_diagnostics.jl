#!/usr/bin/env julia

using CairoMakie
using DataFrames
using Distributions
using Random
using StatsBase
using StatsModels

pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using MultistateModels: Hazard, multistatemodel, set_parameters!, simulate_path, call_haz, call_cumulhaz, survprob, truncate_distribution, CachedTransformStrategy, DirectTransformStrategy
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

# Time-varying covariate configuration: covariate changes at t_changes boundaries
# Using multiple change points to test more complex TVC scenarios
const TVC_CONFIG = Dict(
    "exp" => (; rate = 0.35, beta = 0.6, horizon = 5.0, hazard_start = 0.0, t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
    "wei" => (; rate = 0.0, shape = 1.35, scale = 0.4, beta = -0.35, horizon = 5.0, hazard_start = 0.02, t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
    "gom" => (; rate = 0.0, shape = 0.6, scale = 0.4, beta = 0.5, horizon = 5.0, hazard_start = 0.0, t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
)

struct Scenario
    family::String
    effect::Symbol
    covariate_mode::Symbol  # :baseline, :covariate, or :tvc (time-varying)
    label::String
    slug::String
    config::NamedTuple
end

function Scenario(family::String, effect::Symbol, cov_mode::Symbol)
    if cov_mode == :tvc
        config = TVC_CONFIG[family]
        label = string(uppercasefirst(lowercase(family)), " ", uppercase(String(effect)), " time-varying covariate")
    else
        config = FAMILY_CONFIG[family]
        label = string(uppercasefirst(lowercase(family)), " ", uppercase(String(effect)), " ", cov_mode == :covariate ? "with covariate" : "baseline-only")
    end
    slug = string(family, "_", effect, "_", cov_mode)
    return Scenario(family, effect, cov_mode, label, slug, config)
end

# Include TVC scenarios for exponential and Weibull (PH effect, as AFT requires additional work)
# Gompertz TVC requires more complex piecewise integration and is omitted for now
const SCENARIOS = vcat(
    [Scenario(fam, eff, cov) for fam in keys(FAMILY_CONFIG) for eff in (:ph, :aft) for cov in (:baseline, :covariate)],
    [Scenario(fam, :ph, :tvc) for fam in ["exp", "wei"]]  # TVC scenarios - PH only
)

function scenario_subject_df(scenario::Scenario)
    horizon = scenario.config.horizon
    if scenario.covariate_mode == :tvc
        # Time-varying covariate: multiple intervals with different x values
        t_changes = scenario.config.t_changes
        x_values = scenario.config.x_values
        
        # Build interval boundaries: [0, t_changes..., horizon]
        tstart_grid = vcat(0.0, t_changes)
        tstop_grid = vcat(t_changes, horizon)
        n_intervals = length(tstart_grid)
        
        df = DataFrame(
            id = fill(1, n_intervals),
            tstart = tstart_grid,
            tstop = tstop_grid,
            statefrom = fill(1, n_intervals),
            stateto = fill(2, n_intervals),
            obstype = fill(1, n_intervals),
            x = x_values,
        )
    else
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
    end
    return df
end

function hazard_formula(scenario::Scenario)
    (scenario.covariate_mode == :covariate || scenario.covariate_mode == :tvc) ? @formula(0 ~ x) : @formula(0 ~ 1)
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
    return (scenario.covariate_mode == :covariate || scenario.covariate_mode == :tvc) ? vcat(base, [cfg.beta]) : base
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
    return model, data
end

function covariate_value(scenario::Scenario)
    return scenario.covariate_mode == :covariate ? COVARIATE_VALUE : 0.0
end

# Helper functions for piecewise hazard/cumhaz computation with TVC
function exp_ph_hazard(t, rate, beta, x)
    return rate * exp(beta * x)
end

function exp_ph_cumhaz(t, rate, beta, x)
    return rate * exp(beta * x) * t
end

function wei_ph_hazard(t, shape, scale, beta, x)
    return shape * scale * (t^(shape - 1)) * exp(beta * x)
end

function wei_ph_cumhaz(t, shape, scale, beta, x)
    return scale * exp(beta * x) * (t^shape)
end

# Piecewise cumulative hazard for multiple TVC intervals
function piecewise_exp_ph_cumhaz(t, rate, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += rate * exp(beta * x_values[i]) * (t - prev_t)
            return cumhaz
        else
            cumhaz += rate * exp(beta * x_values[i]) * (tc - prev_t)
            prev_t = tc
        end
    end
    cumhaz += rate * exp(beta * x_values[end]) * (t - prev_t)
    return cumhaz
end

function piecewise_wei_ph_cumhaz(t, shape, scale, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += scale * exp(beta * x_values[i]) * (t^shape - prev_t^shape)
            return cumhaz
        else
            cumhaz += scale * exp(beta * x_values[i]) * (tc^shape - prev_t^shape)
            prev_t = tc
        end
    end
    cumhaz += scale * exp(beta * x_values[end]) * (t^shape - prev_t^shape)
    return cumhaz
end

function expected_curves(scenario::Scenario, times_h::Vector{Float64}, times_cs::Vector{Float64})
    cfg = scenario.config
    
    if scenario.covariate_mode == :tvc
        # Time-varying covariate scenario with multiple change points
        t_changes = cfg.t_changes
        x_values = cfg.x_values
        beta = cfg.beta
        
        if scenario.family == "exp"
            rate = cfg.rate
            # Piecewise hazard - find which interval each time falls in
            function get_x_at_t(t)
                for (i, tc) in enumerate(t_changes)
                    if t < tc
                        return x_values[i]
                    end
                end
                return x_values[end]
            end
            
            haz_expected = [exp_ph_hazard(t, rate, beta, get_x_at_t(t)) for t in times_h]
            
            # Piecewise cumulative hazard
            cum_expected = [piecewise_exp_ph_cumhaz(t, rate, beta, t_changes, x_values) for t in times_cs]
            
        elseif scenario.family == "wei"
            shape = cfg.shape
            scale = cfg.scale
            
            function get_x_at_t_wei(t)
                for (i, tc) in enumerate(t_changes)
                    if t < tc
                        return x_values[i]
                    end
                end
                return x_values[end]
            end
            
            haz_expected = [wei_ph_hazard(t, shape, scale, beta, get_x_at_t_wei(t)) for t in times_h]
            cum_expected = [piecewise_wei_ph_cumhaz(t, shape, scale, beta, t_changes, x_values) for t in times_cs]
        else
            error("TVC not implemented for family $(scenario.family)")
        end
    else
        # Non-TVC scenarios (original code)
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
    end
    surv_expected = exp.(-cum_expected)
    return (; haz_expected, cum_expected, surv_expected)
end

function distribution_functions(scenario::Scenario)
    cfg = scenario.config
    
    if scenario.covariate_mode == :tvc
        # Time-varying covariate - piecewise distribution with multiple intervals
        t_changes = cfg.t_changes
        x_values = cfg.x_values
        beta = cfg.beta
        
        if scenario.family == "exp"
            rate = cfg.rate
            cumhaz = t -> piecewise_exp_ph_cumhaz(t, rate, beta, t_changes, x_values)
            hazard = t -> begin
                # Find which interval t falls into
                for (i, tc) in enumerate(t_changes)
                    if t < tc
                        return rate * exp(beta * x_values[i])
                    end
                end
                return rate * exp(beta * x_values[end])
            end
        elseif scenario.family == "wei"
            shape = cfg.shape
            scale = cfg.scale
            cumhaz = t -> piecewise_wei_ph_cumhaz(t, shape, scale, beta, t_changes, x_values)
            hazard = t -> begin
                for (i, tc) in enumerate(t_changes)
                    if t < tc
                        return shape * scale * exp(beta * x_values[i]) * (t^(shape - 1))
                    end
                end
                return shape * scale * exp(beta * x_values[end]) * (t^(shape - 1))
            end
        else
            error("TVC not implemented for family $(scenario.family)")
        end
    else
        # Non-TVC scenarios (original code)
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
    end
    cdf = t -> t <= 0 ? 0.0 : 1 - exp(-cumhaz(t))
    pdf = t -> t <= 0 ? 0.0 : hazard(t) * exp(-cumhaz(t))
    return cdf, pdf
end

function hazard_time_grid(scenario::Scenario)
    horizon = scenario.config.horizon
    start = get(scenario.config, :hazard_start, 0.0)
    haz_start = start == 0.0 ? 0.0 : start
    return collect(range(haz_start, horizon; length = 200)), collect(range(0.0, horizon; length = 200))
end

function collect_event_durations(model, nsamples; use_cached_strategy::Bool, rng::AbstractRNG)
    durations = Vector{Float64}(undef, nsamples)
    collected = 0
    attempts = 0
    max_attempts = nsamples * 200
    strategy = use_cached_strategy ? CachedTransformStrategy() : DirectTransformStrategy()
    while collected < nsamples
        path = simulate_path(model, 1, DELTA_U, DELTA_T; strategy = strategy, rng = rng)
        attempts += 1
        attempts > max_attempts && error("Exceeded maximum attempts without enough uncensored paths")
        if path.states[end] != path.states[1]
            collected += 1
            durations[collected] = path.times[end] - path.times[1]
        end
    end
    return durations
end

function plot_function_panel(scenario::Scenario, model, data)
    times_h, times_cs = hazard_time_grid(scenario)
    curves = expected_curves(scenario, times_h, times_cs)
    hazard = model.hazards[1]
    pars = model.parameters[1]
    
    if scenario.covariate_mode == :tvc
        # For TVC, use the correct row based on time
        t_change = scenario.config.t_change
        row1 = data[1, :]
        row2 = data[2, :]
        
        # Hazard at each time point uses the appropriate row
        haz_calc = [t < t_change ? call_haz(t, pars, row1, hazard; give_log = false, apply_transform = false) : 
                                   call_haz(t, pars, row2, hazard; give_log = false, apply_transform = false) for t in times_h]
        haz_tt = [t < t_change ? call_haz(t, pars, row1, hazard; give_log = false, apply_transform = true) : 
                                 call_haz(t, pars, row2, hazard; give_log = false, apply_transform = true) for t in times_h]
        
        # Cumulative hazard: piecewise integration
        cum_calc = [begin
            if t <= t_change
                call_cumulhaz(0.0, t, pars, row1, hazard; give_log = false, apply_transform = false)
            else
                call_cumulhaz(0.0, t_change, pars, row1, hazard; give_log = false, apply_transform = false) +
                call_cumulhaz(t_change, t, pars, row2, hazard; give_log = false, apply_transform = false)
            end
        end for t in times_cs]
        cum_tt = [begin
            if t <= t_change
                call_cumulhaz(0.0, t, pars, row1, hazard; give_log = false, apply_transform = true)
            else
                call_cumulhaz(0.0, t_change, pars, row1, hazard; give_log = false, apply_transform = true) +
                call_cumulhaz(t_change, t, pars, row2, hazard; give_log = false, apply_transform = true)
            end
        end for t in times_cs]
        
        # Survival: exp(-cumhaz)
        surv_calc = exp.(-cum_calc)
        surv_tt = exp.(-cum_tt)
    else
        # Non-TVC: use single row
        subj_row = data[1, :]
        haz_calc = [call_haz(t, pars, subj_row, hazard; give_log = false, apply_transform = false) for t in times_h]
        haz_tt = [call_haz(t, pars, subj_row, hazard; give_log = false, apply_transform = true) for t in times_h]
        cum_calc = [call_cumulhaz(0.0, t, pars, subj_row, hazard; give_log = false, apply_transform = false) for t in times_cs]
        cum_tt = [call_cumulhaz(0.0, t, pars, subj_row, hazard; give_log = false, apply_transform = true) for t in times_cs]
        surv_calc = [survprob(0.0, t, model.parameters, subj_row, model.totalhazards[1], model.hazards; give_log = false, apply_transform = false) for t in times_cs]
        surv_tt = [survprob(0.0, t, model.parameters, subj_row, model.totalhazards[1], model.hazards; give_log = false, apply_transform = true) for t in times_cs]
    end

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
    durations_tt = collect_event_durations(model, SIM_SAMPLES; use_cached_strategy = true, rng = rng_tt)
    durations_fb = collect_event_durations(model, SIM_SAMPLES; use_cached_strategy = false, rng = rng_fb)

    ecdf_tt = ecdf(durations_tt)
    ecdf_fb = ecdf(durations_fb)
    horizon = scenario.config.horizon
    ts = collect(range(0.0, horizon; length = DIST_GRID_POINTS))
    cdf_base, pdf_base = distribution_functions(scenario)
    cdf_fn, pdf_fn = truncate_distribution(cdf_base, pdf_base; lower = 0.0, upper = horizon)
    expected = cdf_fn.(ts)
    empirical = ecdf_fb.(ts)
    diff_curve = ecdf_tt.(ts) .- ecdf_fb.(ts)
    max_abs_diff = maximum(abs.(diff_curve))
    ylim_span = max(max_abs_diff, 1e-6)
    xs = ts

    # Compute KS statistic at logarithmically-spaced sample sizes
    # KS_n = max_{i=1:n} |i/n - F(x_{(i)})| where x_{(i)} is the i-th order statistic
    # This should decrease as ~1/√n for correctly distributed samples
    sorted_durations = sort(durations_fb)
    n_samples = length(sorted_durations)
    expected_cdf_at_samples = cdf_fn.(sorted_durations)
    
    # Evaluate KS at specific sample sizes
    eval_ns = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, n_samples]
    eval_ns = filter(n -> n <= n_samples, eval_ns)
    ks_at_n = zeros(length(eval_ns))
    
    for (idx, n) in enumerate(eval_ns)
        # KS statistic for first n samples (order statistics x_{(1)}, ..., x_{(n)})
        # KS_n = max_{i=1:n} max(|i/n - F(x_{(i)})|, |(i-1)/n - F(x_{(i)})|)
        max_diff = 0.0
        for i in 1:n
            cdf_i = expected_cdf_at_samples[i]  # F(x_{(i)})
            # Two-sided KS: check both sides of the step
            diff_upper = abs(i / n - cdf_i)
            diff_lower = abs((i - 1) / n - cdf_i)
            max_diff = max(max_diff, diff_upper, diff_lower)
        end
        ks_at_n[idx] = max_diff
    end

    fig = Figure(size = (1500, 600))
    ax1 = Axis(fig[1, 1], title = "ECDF vs expected", xlabel = "Duration", ylabel = "F(t)")
    lines!(ax1, ts, expected, color = :black, linewidth = 3, label = "analytic")
    lines!(ax1, ts, empirical, color = :dodgerblue, linewidth = 2, label = "simulate_path")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[1, 2], title = "KS statistic vs sample size", xlabel = "Sample size (n)", ylabel = "Dₙ = sup|F̂ₙ − F|", xscale = log10)
    scatterlines!(ax2, eval_ns, ks_at_n, color = :crimson, linewidth = 2, markersize = 8)

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
        model, data = build_model(scenario)
        plot_function_panel(scenario, model, data)
        plot_distribution_panel(scenario, model)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end
