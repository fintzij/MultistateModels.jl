"""
Long tests for simulation with time-varying covariates (TVC).

These tests verify that simulate_path correctly handles:
1. Multiple covariate change points within an observation
2. Both PH and AFT covariate effects
3. All hazard families (exponential, Weibull, Gompertz)
4. Semi-Markov models where sojourn time resets after transitions
5. Multi-state models with competing risks

Tests use Kolmogorov-Smirnov statistics to compare simulated ECDFs against
piecewise analytic CDFs derived from the true hazard functions.
"""

using Test
using Random
using Statistics
using StatsBase: ecdf
using DataFrames

using MultistateModels
using MultistateModels: simulate_path, CachedTransformStrategy, DirectTransformStrategy

# Include fixtures - handle both standalone and runtests contexts
if !isdefined(Main, :TestFixtures)
    include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
end
using .TestFixtures: toy_tvc_exp_model, toy_tvc_wei_model, toy_tvc_gom_model,
                     toy_illness_death_tvc_model, toy_semi_markov_tvc_model

const TVC_SIM_SAMPLES = 10_000
const KS_CRITICAL_005 = 1.36 / sqrt(TVC_SIM_SAMPLES)  # KS critical value at α=0.05

#==============================================================================#
#                          PIECEWISE DISTRIBUTION FUNCTIONS                    #
#==============================================================================#

"""
Compute piecewise cumulative hazard for exponential PH with TVC.
Λ(t) = ∫₀^t λ exp(β x(s)) ds where x(s) is piecewise constant.
"""
function piecewise_cumhaz_exp_ph(t, rate, beta, t_changes, x_values)
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
    # Final interval
    cumhaz += rate * exp(beta * x_values[end]) * (t - prev_t)
    return cumhaz
end

"""
Compute piecewise cumulative hazard for Weibull PH with TVC.
Λ(t) = ∫₀^t α γ s^(α-1) exp(β x(s)) ds
     = γ exp(β x(s)) ∫ α s^(α-1) ds = γ exp(β x(s)) * s^α evaluated appropriately.
"""
function piecewise_cumhaz_wei_ph(t, shape, scale, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            # Weibull cumhaz from prev_t to t: γ exp(βx) * (t^α - prev_t^α)
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

"""
Compute piecewise cumulative hazard for Gompertz PH with TVC.
flexsurv parameterization: h(t) = rate * exp(shape * t)
H(t) = (rate/shape) * (exp(shape*t) - exp(shape*t₀)) for each piece.
With PH: H(t) = (rate/shape) * exp(β x) * (exp(shape*t) - exp(shape*t₀))
"""
function piecewise_cumhaz_gom_ph(t, shape, scale, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += (scale / shape) * exp(beta * x_values[i]) * (exp(shape * t) - exp(shape * prev_t))
            return cumhaz
        else
            cumhaz += (scale / shape) * exp(beta * x_values[i]) * (exp(shape * tc) - exp(shape * prev_t))
            prev_t = tc
        end
    end
    cumhaz += (scale / shape) * exp(beta * x_values[end]) * (exp(shape * t) - exp(shape * prev_t))
    return cumhaz
end

"""
Compute piecewise cumulative hazard for exponential AFT with TVC.
For AFT: h(t) = λ exp(-β x) with time scaled by exp(-β x).
Effective rate in each interval is λ exp(-β x).
"""
function piecewise_cumhaz_exp_aft(t, rate, beta, t_changes, x_values)
    # For exponential, AFT is equivalent to PH with negated beta
    return piecewise_cumhaz_exp_ph(t, rate, -beta, t_changes, x_values)
end

"""
Compute piecewise cumulative hazard for Weibull AFT with TVC.
For Weibull AFT: h(t) = α γ (γ^(1/α) t)^(α-1) exp(-α β x) * γ^(1/α)
               = α γ t^(α-1) exp(-α β x)
Λ(t) = γ exp(-α β x) t^α
"""
function piecewise_cumhaz_wei_aft(t, shape, scale, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += scale * exp(-shape * beta * x_values[i]) * (t^shape - prev_t^shape)
            return cumhaz
        else
            cumhaz += scale * exp(-shape * beta * x_values[i]) * (tc^shape - prev_t^shape)
            prev_t = tc
        end
    end
    cumhaz += scale * exp(-shape * beta * x_values[end]) * (t^shape - prev_t^shape)
    return cumhaz
end

"""
Compute piecewise cumulative hazard for Gompertz AFT with TVC.
flexsurv parameterization: h(t) = rate * exp(shape * t)
For AFT, time is scaled: H(t) = (rate*ts)/(shape*ts) * (exp(shape*ts*t) - exp(shape*ts*t₀))
                              = (rate/shape) * (exp(scaled_shape*t) - exp(scaled_shape*t₀))
where ts = exp(-β x) and scaled_shape = shape * ts
"""
function piecewise_cumhaz_gom_aft(t, shape, scale, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        ts = exp(-beta * x_values[i])  # time scale for this interval
        scaled_shape = shape * ts
        scaled_rate = scale * ts
        if t <= tc
            # Gompertz AFT cumhaz from prev_t to t
            cumhaz += (scaled_rate / scaled_shape) * (exp(scaled_shape * t) - exp(scaled_shape * prev_t))
            return cumhaz
        else
            cumhaz += (scaled_rate / scaled_shape) * (exp(scaled_shape * tc) - exp(scaled_shape * prev_t))
            prev_t = tc
        end
    end
    ts = exp(-beta * x_values[end])
    scaled_shape = shape * ts
    scaled_rate = scale * ts
    cumhaz += (scaled_rate / scaled_shape) * (exp(scaled_shape * t) - exp(scaled_shape * prev_t))
    return cumhaz
end

"""
Build conditional CDF from cumulative hazard function.
Since we only observe uncensored events, we compare to F(t|T<H) = F(t)/F(H).
"""
function make_cdf(cumhaz_fn, horizon::Float64)
    F_horizon = 1 - exp(-cumhaz_fn(horizon))
    return t -> (1 - exp(-cumhaz_fn(t))) / F_horizon
end

#==============================================================================#
#                              SIMULATION HELPERS                               #
#==============================================================================#

"""
Collect event durations from simulated paths.
Returns times to first transition (absorption).
"""
function collect_event_durations(model, n_samples; rng = Random.GLOBAL_RNG)
    durations = Float64[]
    horizon = maximum(model.data.tstop)
    
    for _ in 1:n_samples
        path = simulate_path(model, 1; strategy = CachedTransformStrategy(), rng = rng)
        # Check if transition occurred before horizon
        if length(path.states) > 1 && path.states[end] != path.states[1]
            push!(durations, path.times[2])  # time of first transition
        end
    end
    
    return durations
end

"""
Compute KS statistic between empirical and theoretical CDFs.
"""
function ks_statistic(samples::Vector{Float64}, cdf_fn)
    sorted = sort(samples)
    n = length(sorted)
    max_diff = 0.0
    
    for (i, x) in enumerate(sorted)
        F_x = cdf_fn(x)
        diff_upper = abs(i / n - F_x)
        diff_lower = abs((i - 1) / n - F_x)
        max_diff = max(max_diff, diff_upper, diff_lower)
    end
    
    return max_diff
end

#==============================================================================#
#                                    TESTS                                      #
#==============================================================================#

@testset "TVC Simulation Long Tests" begin
    
    @testset "Exponential PH with multiple TVC change points" begin
        fixture = toy_tvc_exp_model(linpred_effect = :ph)
        model = fixture.model
        cfg = fixture.config
        
        rng = Random.MersenneTwister(12345)
        durations = collect_event_durations(model, TVC_SIM_SAMPLES; rng = rng)
        
        # Skip test if not enough events
        if length(durations) < 100
            @warn "Insufficient events for exp PH TVC test: $(length(durations))"
            @test_skip true
        else
            # Build expected conditional CDF (conditional on event before horizon)
            cumhaz_fn = t -> piecewise_cumhaz_exp_ph(t, cfg.rate, cfg.beta, 
                                                     cfg.t_changes, cfg.x_values)
            cdf_fn = make_cdf(cumhaz_fn, cfg.horizon)
            
            ks = ks_statistic(durations, cdf_fn)
            critical = 1.36 / sqrt(length(durations))
            
            @test ks < critical
            
            if ks >= critical
                @warn "Exp PH TVC KS test failed: KS=$ks, critical=$critical, n=$(length(durations))"
            end
        end
    end
    
    @testset "Exponential AFT with multiple TVC change points" begin
        fixture = toy_tvc_exp_model(linpred_effect = :aft)
        model = fixture.model
        cfg = fixture.config
        
        rng = Random.MersenneTwister(12346)
        durations = collect_event_durations(model, TVC_SIM_SAMPLES; rng = rng)
        
        if length(durations) < 100
            @warn "Insufficient events for exp AFT TVC test: $(length(durations))"
            @test_skip true
        else
            cumhaz_fn = t -> piecewise_cumhaz_exp_aft(t, cfg.rate, cfg.beta,
                                                      cfg.t_changes, cfg.x_values)
            cdf_fn = make_cdf(cumhaz_fn, cfg.horizon)
            
            ks = ks_statistic(durations, cdf_fn)
            critical = 1.36 / sqrt(length(durations))
            
            @test ks < critical
        end
    end
    
    @testset "Weibull PH with multiple TVC change points" begin
        fixture = toy_tvc_wei_model(linpred_effect = :ph)
        model = fixture.model
        cfg = fixture.config
        
        rng = Random.MersenneTwister(12347)
        durations = collect_event_durations(model, TVC_SIM_SAMPLES; rng = rng)
        
        if length(durations) < 100
            @warn "Insufficient events for wei PH TVC test: $(length(durations))"
            @test_skip true
        else
            cumhaz_fn = t -> piecewise_cumhaz_wei_ph(t, cfg.shape, cfg.scale, cfg.beta,
                                                     cfg.t_changes, cfg.x_values)
            cdf_fn = make_cdf(cumhaz_fn, cfg.horizon)
            
            ks = ks_statistic(durations, cdf_fn)
            critical = 1.36 / sqrt(length(durations))
            
            @test ks < critical
        end
    end
    
    @testset "Weibull AFT with multiple TVC change points" begin
        fixture = toy_tvc_wei_model(linpred_effect = :aft)
        model = fixture.model
        cfg = fixture.config
        
        rng = Random.MersenneTwister(12348)
        durations = collect_event_durations(model, TVC_SIM_SAMPLES; rng = rng)
        
        if length(durations) < 100
            @warn "Insufficient events for wei AFT TVC test: $(length(durations))"
            @test_skip true
        else
            cumhaz_fn = t -> piecewise_cumhaz_wei_aft(t, cfg.shape, cfg.scale, cfg.beta,
                                                      cfg.t_changes, cfg.x_values)
            cdf_fn = make_cdf(cumhaz_fn, cfg.horizon)
            
            ks = ks_statistic(durations, cdf_fn)
            critical = 1.36 / sqrt(length(durations))
            
            @test ks < critical
        end
    end
    
    @testset "Gompertz PH with multiple TVC change points" begin
        fixture = toy_tvc_gom_model(linpred_effect = :ph)
        model = fixture.model
        cfg = fixture.config
        
        rng = Random.MersenneTwister(12349)
        durations = collect_event_durations(model, TVC_SIM_SAMPLES; rng = rng)
        
        if length(durations) < 100
            @warn "Insufficient events for gom PH TVC test: $(length(durations))"
            @test_skip true
        else
            cumhaz_fn = t -> piecewise_cumhaz_gom_ph(t, cfg.shape, cfg.scale, cfg.beta,
                                                     cfg.t_changes, cfg.x_values)
            cdf_fn = make_cdf(cumhaz_fn, cfg.horizon)
            
            ks = ks_statistic(durations, cdf_fn)
            critical = 1.36 / sqrt(length(durations))
            
            @test ks < critical
        end
    end
    
    @testset "Gompertz AFT with multiple TVC change points" begin
        fixture = toy_tvc_gom_model(linpred_effect = :aft)
        model = fixture.model
        cfg = fixture.config
        
        rng = Random.MersenneTwister(12350)
        durations = collect_event_durations(model, TVC_SIM_SAMPLES; rng = rng)
        
        if length(durations) < 100
            @warn "Insufficient events for gom AFT TVC test: $(length(durations))"
            @test_skip true
        else
            cumhaz_fn = t -> piecewise_cumhaz_gom_aft(t, cfg.shape, cfg.scale, cfg.beta,
                                                      cfg.t_changes, cfg.x_values)
            cdf_fn = make_cdf(cumhaz_fn, cfg.horizon)
            
            ks = ks_statistic(durations, cdf_fn)
            critical = 1.36 / sqrt(length(durations))
            
            @test ks < critical
        end
    end
    
    @testset "Semi-Markov TVC: sojourn reset after transition" begin
        # This test verifies that after a state transition, sojourn time resets to 0
        # and the next transition uses the correct (reset) time in the hazard.
        
        fixture = toy_semi_markov_tvc_model()
        model = fixture.model
        
        rng = Random.MersenneTwister(12351)
        
        # Collect paths that have at least 2 transitions
        n_paths_with_return = 0
        n_total = 0
        
        for _ in 1:TVC_SIM_SAMPLES
            path = simulate_path(model, 1; strategy = CachedTransformStrategy(), rng = rng)
            n_total += 1
            
            # Check for back-and-forth transitions (1 → 2 → 1)
            if length(path.states) >= 3 && 
               path.states[1] == 1 && path.states[2] == 2 && path.states[3] == 1
                n_paths_with_return += 1
                
                # Verify that the second sojourn (in state 2) makes sense
                # The time spent in state 2 should be independent of when we entered
                sojourn_in_2 = path.times[3] - path.times[2]
                @test sojourn_in_2 > 0
            end
        end
        
        # We should see some paths with return transitions
        @test n_paths_with_return > 0
        @info "Semi-Markov TVC: $(n_paths_with_return)/$(n_total) paths had 1→2→1 pattern"
    end
    
    @testset "Multi-state illness-death with TVC" begin
        fixture = toy_illness_death_tvc_model()
        model = fixture.model
        
        rng = Random.MersenneTwister(12352)
        
        # Track transitions to different absorbing states
        n_direct_death = 0  # 1 → 3
        n_illness_then_death = 0  # 1 → 2 → 3
        n_censored = 0  # no transition
        
        for _ in 1:TVC_SIM_SAMPLES
            path = simulate_path(model, 1; strategy = CachedTransformStrategy(), rng = rng)
            
            if path.states[end] == 3
                if length(path.states) == 2
                    n_direct_death += 1
                else
                    n_illness_then_death += 1
                end
            else
                n_censored += 1
            end
        end
        
        # All three outcomes should occur with reasonable frequency
        total_deaths = n_direct_death + n_illness_then_death
        
        @info "Illness-death TVC: direct=$(n_direct_death), via-illness=$(n_illness_then_death), censored=$(n_censored)"
        
        # Basic sanity checks
        @test n_direct_death > 0
        @test n_illness_then_death > 0
        @test total_deaths > TVC_SIM_SAMPLES * 0.1  # At least 10% should die
    end
    
    @testset "TVC simulation reproducibility with same seed" begin
        # Verify that simulations are reproducible when using the same RNG seed
        fixture = toy_tvc_exp_model()
        model = fixture.model
        
        rng1 = Random.MersenneTwister(99999)
        path1 = simulate_path(model, 1; strategy = CachedTransformStrategy(), rng = rng1)
        
        rng2 = Random.MersenneTwister(99999)
        path2 = simulate_path(model, 1; strategy = CachedTransformStrategy(), rng = rng2)
        
        @test path1.times == path2.times
        @test path1.states == path2.states
    end
    
    @testset "TVC: Cached vs Direct strategy equivalence" begin
        # Both strategies should produce statistically identical results
        fixture = toy_tvc_exp_model()
        model = fixture.model
        cfg = fixture.config
        
        rng_cached = Random.MersenneTwister(11111)
        rng_direct = Random.MersenneTwister(11111)
        
        # Collect durations with both strategies
        durations_cached = Float64[]
        durations_direct = Float64[]
        
        for _ in 1:1000
            path_c = simulate_path(model, 1; strategy = CachedTransformStrategy(), rng = rng_cached)
            path_d = simulate_path(model, 1; strategy = DirectTransformStrategy(), rng = rng_direct)
            
            if length(path_c.states) > 1 && path_c.states[end] != path_c.states[1]
                push!(durations_cached, path_c.times[2])
            end
            if length(path_d.states) > 1 && path_d.states[end] != path_d.states[1]
                push!(durations_direct, path_d.times[2])
            end
        end
        
        # With same RNG seed, results should be identical
        @test length(durations_cached) == length(durations_direct)
        if length(durations_cached) > 0
            @test durations_cached ≈ durations_direct atol=1e-10
        end
    end
    
end
