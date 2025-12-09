# =============================================================================
# Parameter Ordering Tests
# =============================================================================
#
# These tests verify that parameters are stored, retrieved, and used correctly
# throughout the package. They check:
# 1. Parameter ordering in get_parameters_flat (transition matrix order)
# 2. set_parameters! correctly assigns values via hazard names
# 3. Simulation uses correct parameters
# 4. Fitting recovers true parameters
# 5. Fitted objects store parameters correctly

using Test
using DataFrames
using Random
using Statistics
using MultistateModels
using MultistateModels: get_parameters_flat, set_parameters!, get_estimation_scale_params
using StatsModels: @formula

@testset "Parameter Ordering" begin

    # =========================================================================
    # TEST 1: Basic parameter ordering in 3-state illness-death model
    # =========================================================================
    @testset "3-state model parameter ordering" begin
        # Setup
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # healthy -> ill
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # healthy -> dead
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)  # ill -> dead
        
        N = 10
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(10.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12, h13, h23; data=template)
        
        # Set distinct parameter values
        log_h12 = -1.0  # log(rate) for 1->2
        log_h13 = -2.0  # log(rate) for 1->3
        log_h23 = -3.0  # log(rate) for 2->3
        
        set_parameters!(model, (h12=[log_h12], h13=[log_h13], h23=[log_h23]))
        
        # Verify get_parameters_flat returns in transition matrix order
        flat = get_parameters_flat(model)
        @test length(flat) == 3
        @test flat[1] ≈ log_h12  # h12 first (1->2)
        @test flat[2] ≈ log_h13  # h13 second (1->3)
        @test flat[3] ≈ log_h23  # h23 third (2->3)
        
        # Verify hazard names match expected order
        @test model.hazards[1].hazname == :h12
        @test model.hazards[2].hazname == :h13
        @test model.hazards[3].hazname == :h23
    end
    
    # =========================================================================
    # TEST 2: Weibull parameter ordering (shape, scale)
    # =========================================================================
    @testset "Weibull parameter ordering" begin
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        N = 10
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(10.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12; data=template)
        
        # Set parameters: [log_shape, log_scale]
        log_shape = log(1.5)
        log_scale = log(0.1)
        set_parameters!(model, (h12=[log_shape, log_scale],))
        
        flat = get_parameters_flat(model)
        @test length(flat) == 2
        @test flat[1] ≈ log_shape  # First parameter is shape
        @test flat[2] ≈ log_scale  # Second parameter is scale
        
        # Verify parameter names
        @test model.hazards[1].parnames == [:h12_shape, :h12_scale]
    end
    
    # =========================================================================
    # TEST 3: Gompertz parameter ordering (shape, scale)
    # =========================================================================
    @testset "Gompertz parameter ordering" begin
        h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
        
        N = 10
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(10.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12; data=template)
        
        # Set parameters: [log_shape, log_scale]
        log_shape = log(0.3)
        log_scale = log(0.05)
        set_parameters!(model, (h12=[log_shape, log_scale],))
        
        flat = get_parameters_flat(model)
        @test length(flat) == 2
        @test flat[1] ≈ log_shape  # First parameter is shape
        @test flat[2] ≈ log_scale  # Second parameter is scale
        
        # Verify parameter names
        @test model.hazards[1].parnames == [:h12_shape, :h12_scale]
    end
    
    # =========================================================================
    # TEST 4: Mixed hazards model ordering
    # =========================================================================
    @testset "Mixed hazard types ordering" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # 1 param
        h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)  # 2 params
        h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)  # 2 params
        
        N = 10
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(10.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12, h13, h23; data=template)
        
        # Set parameters
        log_rate = log(0.2)
        log_wei_shape = log(1.5)
        log_wei_scale = log(0.1)
        log_gom_shape = log(0.3)
        log_gom_scale = log(0.05)
        
        set_parameters!(model, (
            h12=[log_rate], 
            h13=[log_wei_shape, log_wei_scale],
            h23=[log_gom_shape, log_gom_scale]
        ))
        
        flat = get_parameters_flat(model)
        @test length(flat) == 5  # 1 + 2 + 2
        
        # Check ordering: h12 (1 param), h13 (2 params), h23 (2 params)
        @test flat[1] ≈ log_rate         # h12 rate
        @test flat[2] ≈ log_wei_shape    # h13 shape
        @test flat[3] ≈ log_wei_scale    # h13 scale
        @test flat[4] ≈ log_gom_shape    # h23 shape
        @test flat[5] ≈ log_gom_scale    # h23 scale
    end

end

@testset "Simulation Parameter Usage" begin

    # =========================================================================
    # TEST 5: Exponential simulation uses correct rate
    # =========================================================================
    @testset "Exponential simulation" begin
        rate = 0.2
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        N = 5000
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12; data=template)
        set_parameters!(model, (h12=[log(rate)],))
        
        sim = simulate(model; paths=false, data=true, nsim=1)
        times = sim[1].tstop .- sim[1].tstart
        observed_mean = mean(times)
        expected_mean = 1 / rate  # E[T] = 1/λ for exponential
        
        relative_error = abs(observed_mean - expected_mean) / expected_mean
        @test relative_error < 0.05  # Less than 5% error
    end
    
    # =========================================================================
    # TEST 6: Weibull simulation uses correct shape and scale
    # =========================================================================
    @testset "Weibull simulation" begin
        shape = 1.5
        scale = 0.1
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        N = 5000
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12; data=template)
        set_parameters!(model, (h12=[log(shape), log(scale)],))
        
        sim = simulate(model; paths=false, data=true, nsim=1)
        times = sim[1].tstop .- sim[1].tstart
        observed_mean = mean(times)
        
        # Expected mean for Weibull: (1/scale)^(1/shape) * Γ(1 + 1/shape)
        using SpecialFunctions
        expected_mean = (1/scale)^(1/shape) * gamma(1 + 1/shape)
        
        relative_error = abs(observed_mean - expected_mean) / expected_mean
        @test relative_error < 0.05  # Less than 5% error
    end
    
    # =========================================================================
    # TEST 7: Gompertz simulation uses correct shape and rate
    # NOTE: Gompertz uses flexsurv parameterization where:
    #   - shape is UNCONSTRAINED (identity transform on estimation scale)
    #   - rate is POSITIVE (log-transformed on estimation scale)
    # =========================================================================
    @testset "Gompertz simulation" begin
        shape = 0.3
        rate = 0.05
        
        h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
        
        N = 5000
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model = multistatemodel(h12; data=template)
        # Shape is on natural scale (identity transform), rate is log-transformed
        set_parameters!(model, (h12=[shape, log(rate)],))
        
        sim = simulate(model; paths=false, data=true, nsim=1)
        times = sim[1].tstop .- sim[1].tstart
        observed_mean = mean(times)
        
        # Expected mean: ∫₀^∞ S(t) dt where S(t) = exp(-H(t))
        # flexsurv parameterization: H(t) = (rate/shape) * (exp(shape*t) - 1)
        using QuadGK
        surv(t) = exp(-(rate / shape) * (exp(shape * t) - 1))
        expected_mean, _ = quadgk(surv, 0, 100)
        
        relative_error = abs(observed_mean - expected_mean) / expected_mean
        @test relative_error < 0.05  # Less than 5% error
    end

end

@testset "Parameter Recovery via Fitting" begin

    # =========================================================================
    # TEST 8: Exponential parameter recovery
    # =========================================================================
    @testset "Exponential fitting" begin
        true_rate = 0.15
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        N = 500
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model_sim = multistatemodel(h12; data=template)
        set_parameters!(model_sim, (h12=[log(true_rate)],))
        
        sim = simulate(model_sim; paths=false, data=true, nsim=1)
        data = sim[1]
        
        model_fit = multistatemodel(h12; data=data)
        fitted = fit(model_fit; parallel=true, verbose=false)
        
        est_rate = exp(get_parameters_flat(fitted)[1])
        relative_error = abs(est_rate - true_rate) / true_rate
        
        @test relative_error < 0.15  # Less than 15% error with N=500
    end
    
    # =========================================================================
    # TEST 9: Weibull parameter recovery
    # =========================================================================
    @testset "Weibull fitting" begin
        true_shape = 1.5
        true_scale = 0.1
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        N = 500
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model_sim = multistatemodel(h12; data=template)
        set_parameters!(model_sim, (h12=[log(true_shape), log(true_scale)],))
        
        sim = simulate(model_sim; paths=false, data=true, nsim=1)
        data = sim[1]
        
        model_fit = multistatemodel(h12; data=data)
        fitted = fit(model_fit; parallel=true, verbose=false)
        
        flat = get_parameters_flat(fitted)
        est_shape = exp(flat[1])
        est_scale = exp(flat[2])
        
        shape_error = abs(est_shape - true_shape) / true_shape
        scale_error = abs(est_scale - true_scale) / true_scale
        
        @test shape_error < 0.15  # Less than 15% error
        @test scale_error < 0.15  # Less than 15% error
    end
    
    # =========================================================================
    # TEST 10: 3-state exponential model parameter recovery
    # =========================================================================
    @testset "3-state model fitting" begin
        true_h12 = 0.15
        true_h13 = 0.05
        true_h23 = 0.20
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        N = 1000
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model_sim = multistatemodel(h12, h13, h23; data=template)
        set_parameters!(model_sim, (
            h12=[log(true_h12)], 
            h13=[log(true_h13)], 
            h23=[log(true_h23)]
        ))
        
        sim = simulate(model_sim; paths=false, data=true, nsim=1)
        data = sim[1]
        
        model_fit = multistatemodel(h12, h13, h23; data=data)
        fitted = fit(model_fit; parallel=true, verbose=false)
        
        flat = get_parameters_flat(fitted)
        est_h12 = exp(flat[1])
        est_h13 = exp(flat[2])
        est_h23 = exp(flat[3])
        
        err_h12 = abs(est_h12 - true_h12) / true_h12
        err_h13 = abs(est_h13 - true_h13) / true_h13
        err_h23 = abs(est_h23 - true_h23) / true_h23
        
        @test err_h12 < 0.15  # Less than 15% error
        @test err_h13 < 0.20  # Larger tolerance for less common transition
        @test err_h23 < 0.15  # Less than 15% error
    end

end

@testset "Fitted Object Parameter Storage" begin

    # =========================================================================
    # TEST 11: Verify fitted parameters match internal storage
    # =========================================================================
    @testset "Fitted parameter consistency" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        N = 200
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        model_sim = multistatemodel(h12; data=template)
        set_parameters!(model_sim, (h12=[log(0.15)],))
        
        sim = simulate(model_sim; paths=false, data=true, nsim=1)
        data = sim[1]
        
        model_fit = multistatemodel(h12; data=data)
        fitted = fit(model_fit; parallel=true, verbose=false)
        
        # Compare get_parameters_flat to internal storage
        flat = get_parameters_flat(fitted)
        internal = get_estimation_scale_params(fitted.parameters)
        
        # internal is now a NamedTuple by hazard name
        # Get the first hazard's baseline parameters
        first_hazard = first(values(internal))
        first_baseline_vals = collect(values(first_hazard.baseline))
        @test flat[1] ≈ first_baseline_vals[1]
    end
    
    # =========================================================================
    # TEST 12: Verify re-simulation with fitted parameters
    # =========================================================================
    @testset "Re-simulation with fitted parameters" begin
        true_rate = 0.2
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        N = 1000
        Random.seed!(12345)
        template = DataFrame(
            id = 1:N,
            tstart = zeros(N),
            tstop = fill(100.0, N),
            statefrom = ones(Int, N),
            stateto = ones(Int, N),
            obstype = ones(Int, N)
        )
        
        # Simulate from true model
        model_sim = multistatemodel(h12; data=template)
        set_parameters!(model_sim, (h12=[log(true_rate)],))
        sim1 = simulate(model_sim; paths=false, data=true, nsim=1)
        data = sim1[1]
        
        # Fit the model
        model_fit = multistatemodel(h12; data=data)
        fitted = fit(model_fit; parallel=true, verbose=false)
        
        # Re-simulate from fitted model using same template
        est_rate = exp(get_parameters_flat(fitted)[1])
        model_resim = multistatemodel(h12; data=template)
        set_parameters!(model_resim, (h12=[log(est_rate)],))
        
        Random.seed!(54321)
        sim2 = simulate(model_resim; paths=false, data=true, nsim=1)
        times = sim2[1].tstop .- sim2[1].tstart
        observed_mean = mean(times)
        expected_mean = 1 / est_rate
        
        relative_error = abs(observed_mean - expected_mean) / expected_mean
        @test relative_error < 0.05  # Less than 5% error
    end

end
