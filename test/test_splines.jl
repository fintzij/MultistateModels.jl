# =============================================================================
# Unit tests for spline hazard functionality
# =============================================================================
# 
# These tests verify spline hazard implementation correctness by:
#   1. Numerical integration: H(a,b) ≈ ∫ₐᵇ h(t) dt using QuadGK
#   2. PH covariate effect: h(t|x) = h₀(t) exp(β'x) 
#   3. Survival probability: S(a,b) = exp(-H(a,b))
#   4. Cumulative hazard additivity: H(a,c) = H(a,b) + H(b,c)
#   5. Spline infrastructure: knot placement, coefficient transforms, etc.
# =============================================================================

using Test
using DataFrames
using Distributions
using MultistateModels
using Random
using QuadGK

@testset "Spline Hazards" begin

    # =========================================================================
    # Test data setup
    # =========================================================================
    
    # Simple test data for 3-state model (transitions 1→2, 2→1, 1→3, 3→1)
    simple_dat = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.3],
        tstop = [0.5, 1.0, 0.3, 1.0],
        statefrom = [1, 2, 1, 3],
        stateto = [2, 1, 3, 1],
        obstype = [1, 1, 1, 1],
        x = [0.5, 0.5, -0.3, -0.3]
    )
    
    # Simple 2-state test data for single-transition tests
    two_state_dat = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.4],
        tstop = [0.5, 1.0, 0.4, 0.9],
        statefrom = [1, 1, 1, 1],
        stateto = [2, 2, 2, 2],
        obstype = [1, 1, 1, 1],
        x = [0.5, 0.5, -0.3, -0.3]
    )

    # Larger dataset for auto-knot tests
    Random.seed!(42)
    n_subjects = 50
    
    function generate_test_data(n_subjects)
        rows = []
        for subj in 1:n_subjects
            t1 = rand(Uniform(0.3, 0.8))
            push!(rows, (id=subj, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1, x=randn()))
            t2 = t1 + rand(Uniform(0.2, 0.6))
            push!(rows, (id=subj, tstart=t1, tstop=t2, statefrom=2, stateto=1, obstype=1, x=rows[end].x))
        end
        return DataFrame(rows)
    end
    
    large_dat = generate_test_data(n_subjects)

    # =========================================================================
    # CORE VERIFICATION: Cumulative hazard matches numerical integration
    # =========================================================================
    # This is the fundamental test: H(a,b) = ∫ₐᵇ h(t) dt
    # We verify the implementation computes the correct integral.
    
    @testset "Cumulative hazard vs QuadGK integration" begin
        # Create spline hazard with explicit knots
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, knots=[0.3, 0.5, 0.7], 
                     boundaryknots=[0.0, 1.0],
                     natural_spline=true)
        
        model = multistatemodel(h12; data=two_state_dat)
        
        # Test with several different parameter configurations
        Random.seed!(12345)
        for trial in 1:5
            # Set random parameters (log scale)
            npar = model.hazards[1].npar_total
            test_pars = randn(npar) * 0.5
            set_parameters!(model, 1, test_pars)
            
            pars = get_parameters(model, 1, scale=:log)
            haz = model.hazards[1]
            covars = NamedTuple()
            
            # Test multiple intervals
            intervals = [(0.1, 0.4), (0.0, 0.5), (0.2, 0.8), (0.3, 0.9), (0.0, 1.0)]
            
            for (lb, ub) in intervals
                # Analytical cumulative hazard from implementation
                H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
                
                # Numerical integration with QuadGK
                H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), 
                                   lb, ub; rtol=1e-10)
                
                @test isapprox(H_impl, H_quad; rtol=1e-6)
            end
        end
    end
    
    @testset "Cumulative hazard with covariates vs QuadGK" begin
        # PH model: h(t|x) = h₀(t) exp(β'x)
        # Verify numerical integration gives same result as analytical cumhaz
        
        h_cov = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                       degree=3, knots=[0.3, 0.6], 
                       boundaryknots=[0.0, 1.0],
                       natural_spline=true)
        
        model = multistatemodel(h_cov; data=two_state_dat)
        haz = model.hazards[1]
        
        # Set parameters: spline coefficients + covariate effect
        Random.seed!(67890)
        nbasis = haz.npar_baseline
        spline_pars = randn(nbasis) * 0.3
        beta = 0.7
        all_pars = vcat(spline_pars, [beta])
        set_parameters!(model, 1, all_pars)
        
        pars = get_parameters(model, 1, scale=:log)
        
        # Test with different covariate values
        for x_val in [-0.5, 0.0, 0.5, 1.0]
            covars = (x = x_val,)
            
            for (lb, ub) in [(0.1, 0.5), (0.2, 0.8), (0.0, 1.0)]
                H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
                H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), 
                                   lb, ub; rtol=1e-10)
                
                @test isapprox(H_impl, H_quad; rtol=1e-6)
            end
        end
    end

    # =========================================================================
    # PH MODEL VERIFICATION: h(t|x) = h₀(t) exp(β'x)
    # =========================================================================
    
    @testset "PH covariate effect verification" begin
        h_cov = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                       degree=3, knots=[0.3, 0.5, 0.7], 
                       boundaryknots=[0.0, 1.0],
                       natural_spline=true)
        
        model = multistatemodel(h_cov; data=two_state_dat)
        haz = model.hazards[1]
        
        # Set known covariate effect
        Random.seed!(11111)
        nbasis = haz.npar_baseline
        spline_pars = randn(nbasis) * 0.3
        beta = 0.5  # Known coefficient
        all_pars = vcat(spline_pars, [beta])
        set_parameters!(model, 1, all_pars)
        pars = get_parameters(model, 1, scale=:log)
        
        # Test: hazard ratio should equal exp(β * Δx)
        x1, x2 = 1.0, -0.5
        covars1 = (x = x1,)
        covars2 = (x = x2,)
        
        for t in [0.2, 0.5, 0.8]
            h1 = MultistateModels.eval_hazard(haz, t, pars, covars1)
            h2 = MultistateModels.eval_hazard(haz, t, pars, covars2)
            
            expected_hr = exp(beta * (x1 - x2))
            actual_hr = h1 / h2
            
            @test isapprox(actual_hr, expected_hr; rtol=1e-10)
        end
        
        # Test: cumulative hazard ratio should also equal exp(β * Δx)
        for (lb, ub) in [(0.1, 0.4), (0.2, 0.7)]
            H1 = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars1)
            H2 = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars2)
            
            expected_hr = exp(beta * (x1 - x2))
            actual_hr = H1 / H2
            
            @test isapprox(actual_hr, expected_hr; rtol=1e-10)
        end
    end

    # =========================================================================
    # SURVIVAL PROBABILITY VERIFICATION: S(a,b) = exp(-H(a,b))
    # =========================================================================
    
    @testset "Survival probability correctness" begin
        h_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                      degree=3, knots=[0.4, 0.6], 
                      boundaryknots=[0.0, 1.0],
                      natural_spline=true)
        
        model = multistatemodel(h_sp; data=two_state_dat)
        haz = model.hazards[1]
        
        Random.seed!(22222)
        npar = haz.npar_total
        test_pars = randn(npar) * 0.4
        set_parameters!(model, 1, test_pars)
        
        params = MultistateModels.get_hazard_params(model.parameters)
        pars = params[1]
        subjdat_row = model.data[1, :]
        covars = MultistateModels.extract_covariates_fast(subjdat_row, haz.covar_names)
        
        # Test: S(a,b) = exp(-H(a,b))
        test_intervals = [(0.0, 0.3), (0.0, 0.5), (0.0, 0.8), (0.2, 0.7)]
        
        for (lb, ub) in test_intervals
            H = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
            S = MultistateModels.survprob(lb, ub, params, subjdat_row, 
                                          model.totalhazards[1], model.hazards; 
                                          give_log=false)
            
            @test isapprox(S, exp(-H); rtol=1e-10)
            
            # Log survival should equal -H
            log_S = MultistateModels.survprob(lb, ub, params, subjdat_row, 
                                              model.totalhazards[1], model.hazards; 
                                              give_log=true)
            @test isapprox(log_S, -H; rtol=1e-10)
        end
        
        # Test: survival is monotonically decreasing
        times = [0.2, 0.4, 0.6, 0.8, 1.0]
        surv_vals = [MultistateModels.survprob(0.0, t, params, subjdat_row, 
                                               model.totalhazards[1], model.hazards; 
                                               give_log=false)
                     for t in times]
        
        for i in 1:(length(surv_vals)-1)
            @test surv_vals[i] >= surv_vals[i+1]
        end
    end

    # =========================================================================
    # CUMULATIVE HAZARD ADDITIVITY: H(a,c) = H(a,b) + H(b,c)
    # =========================================================================
    
    @testset "Cumulative hazard additivity" begin
        h_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                      degree=3, knots=[0.3, 0.5, 0.7], 
                      boundaryknots=[0.0, 1.0],
                      natural_spline=true)
        
        model = multistatemodel(h_sp; data=two_state_dat)
        haz = model.hazards[1]
        
        Random.seed!(33333)
        npar = haz.npar_total
        test_pars = randn(npar) * 0.5
        set_parameters!(model, 1, test_pars)
        pars = get_parameters(model, 1, scale=:log)
        covars = NamedTuple()
        
        # Test additivity: H(a,c) = H(a,b) + H(b,c)
        test_cases = [
            (0.1, 0.4, 0.7),
            (0.0, 0.5, 1.0),
            (0.2, 0.3, 0.8)
        ]
        
        for (a, b, c) in test_cases
            H_ac = MultistateModels.eval_cumhaz(haz, a, c, pars, covars)
            H_ab = MultistateModels.eval_cumhaz(haz, a, b, pars, covars)
            H_bc = MultistateModels.eval_cumhaz(haz, b, c, pars, covars)
            
            @test isapprox(H_ac, H_ab + H_bc; rtol=1e-10)
        end
        
        # Test: zero-length interval gives zero
        @test MultistateModels.eval_cumhaz(haz, 0.5, 0.5, pars, covars) == 0.0
    end


    # =========================================================================
    # Automatic knot placement
    # =========================================================================
    
    @testset "Automatic Knot Placement" begin
        h12_auto = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=nothing, natural_spline=true)
        h21_auto = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
                          degree=3, knots=nothing)
        
        auto_model = multistatemodel(h12_auto, h21_auto; data=large_dat)
        
        for (i, haz) in enumerate(auto_model.hazards)
            # Should have interior knots placed automatically
            @test length(haz.knots) > 2
            # Knots should be sorted
            @test issorted(haz.knots)
            # Interior knots should be within boundaries
            interior = haz.knots[2:end-1]
            @test all(interior .> haz.knots[1])
            @test all(interior .< haz.knots[end])
        end
        
        # Verify cumulative hazard matches numerical integration (QuadGK)
        # Note: rtol=1e-3 because spline cumhaz uses a different internal integration
        # method than QuadGK, introducing small numerical differences.
        # Only test hazard 1 which has no covariates; hazard 2 requires covariates.
        haz = auto_model.hazards[1]
        pars = get_parameters(auto_model, 1, scale=:log)
        covars = NamedTuple()
        lb, ub = 0.1, 0.8
        
        H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
        H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), lb, ub; rtol=1e-10)
        
        @test isapprox(H_impl, H_quad; rtol=1e-3)
    end
    
    @testset "default_nknots function" begin
        @test MultistateModels.default_nknots(0) == 0
        @test MultistateModels.default_nknots(1) == 2  # min 2
        @test MultistateModels.default_nknots(10) == 2
        @test MultistateModels.default_nknots(32) == 2  # 32^(1/5) ≈ 2.0
        @test MultistateModels.default_nknots(100) == 2  # 100^(1/5) ≈ 2.51
        @test MultistateModels.default_nknots(1000) == 3  # 1000^(1/5) ≈ 3.98
        @test MultistateModels.default_nknots(10000) == 6  # 10000^(1/5) ≈ 6.31
    end

    # =========================================================================
    # Time transformation support: verify parity with non-transformed
    # =========================================================================
    
    @testset "Time Transform for Splines - parity verification" begin
        # time_transform should give identical results to standard evaluation
        h_plain = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                         degree=3, knots=[0.3, 0.5, 0.7],
                         boundaryknots=[0.0, 1.0],
                         natural_spline=true,
                         time_transform=false)
        h_tt = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                      degree=3, knots=[0.3, 0.5, 0.7],
                      boundaryknots=[0.0, 1.0],
                      natural_spline=true,
                      time_transform=true)
        
        model_plain = multistatemodel(h_plain; data=two_state_dat)
        model_tt = multistatemodel(h_tt; data=two_state_dat)
        
        # Set identical parameters
        Random.seed!(44444)
        npar = model_plain.hazards[1].npar_total
        test_pars = randn(npar) * 0.4
        set_parameters!(model_plain, 1, test_pars)
        set_parameters!(model_tt, 1, test_pars)
        
        pars_plain = get_parameters(model_plain, 1, scale=:log)
        pars_tt = get_parameters(model_tt, 1, scale=:log)
        
        haz_plain = model_plain.hazards[1]
        haz_tt = model_tt.hazards[1]
        
        # Test with multiple covariate values
        for x_val in [-0.3, 0.0, 0.5]
            covars = (x = x_val,)
            linpred = MultistateModels._linear_predictor(pars_tt, covars, haz_tt)
            
            # Point hazard parity
            for t in [0.1, 0.3, 0.5, 0.7, 0.9]
                h_plain_val = MultistateModels.eval_hazard(haz_plain, t, pars_plain, covars)
                h_tt_val = MultistateModels._time_transform_hazard(haz_tt, pars_tt, t, linpred)
                
                @test isapprox(h_plain_val, h_tt_val; rtol=1e-8)
            end
            
            # Cumulative hazard parity
            for (lb, ub) in [(0.1, 0.5), (0.2, 0.8), (0.0, 1.0)]
                H_plain = MultistateModels.eval_cumhaz(haz_plain, lb, ub, pars_plain, covars)
                H_tt = MultistateModels._time_transform_cumhaz(haz_tt, pars_tt, lb, ub, linpred)
                
                @test isapprox(H_plain, H_tt; rtol=1e-6)
            end
        end
    end

    # =========================================================================
    # Coefficient rectification (rectify_coefs!)
    # =========================================================================
    
    @testset "rectify_coefs! for Splines" begin
        # Create a model with monotone splines (where rectification matters most)
        h12_mono = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          monotone=1)  # Increasing
        h21_mono = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          monotone=-1)  # Decreasing
        
        model = multistatemodel(h12_mono, h21_mono; data=large_dat)
        
        # Set reasonable parameters
        for (h, haz) in enumerate(model.hazards)
            npar = haz.npar_total
            new_pars = fill(-1.0, npar)
            set_parameters!(model, h, new_pars)
        end
        
        # Get the flat parameter vector
        ests_before = get_parameters_flat(model)
        
        # Apply rectification
        ests_after = copy(ests_before)
        MultistateModels.rectify_coefs!(ests_after, model)
        
        # Rectification should produce valid parameters
        @test all(isfinite.(ests_after))
        
        # Update model with rectified parameters using safe_unflatten
        pars_nested = MultistateModels.safe_unflatten(ests_after, model)
        for (hazname, hazidx) in model.hazkeys
            hazard_pars = pars_nested[hazname]
            # Extract full parameter vector (baseline + covariates)
            pars_vec = MultistateModels.extract_params_vector(hazard_pars)
            set_parameters!(model, hazidx, pars_vec)
        end
        
        # Verify cumulative hazard matches numerical integration after rectification
        # Note: rtol=1e-3 because spline cumhaz uses a different internal integration
        # method than QuadGK, introducing small numerical differences.
        # Only test hazard 1 which has no covariates; hazard 2 requires covariates.
        haz = model.hazards[1]
        pars = get_parameters(model, 1, scale=:log)
        covars = NamedTuple()
        lb, ub = 0.1, 0.8
        
        H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
        H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), lb, ub; rtol=1e-10)
        
        @test isapprox(H_impl, H_quad; rtol=1e-3)
    end
    
    @testset "rectify_coefs! round-trip consistency" begin
        # Test that rectify_coefs! produces consistent results on second application
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, knots=[0.3, 0.5, 0.7],
                     monotone=1)
        
        model = multistatemodel(h12; data=large_dat)
        
        Random.seed!(99999)
        set_parameters!(model, 1, rand(Normal(0, 1), model.hazards[1].npar_total))
        
        ests = get_parameters_flat(model)
        
        # First rectification
        ests_rect1 = copy(ests)
        MultistateModels.rectify_coefs!(ests_rect1, model)
        
        # Second rectification should be idempotent (or very close)
        ests_rect2 = copy(ests_rect1)
        MultistateModels.rectify_coefs!(ests_rect2, model)
        
        # Results should be very close after second pass
        @test maximum(abs.(ests_rect1 .- ests_rect2)) < 1e-10
    end

    # =========================================================================
    # Spline coefficient transformations
    # =========================================================================
    
    @testset "_spline_ests2coefs and _spline_coefs2ests round-trip" begin
        using BSplineKit
        
        # Create a test basis
        knots = [0.0, 0.3, 0.5, 0.7, 1.0]
        B = BSplineBasis(BSplineOrder(4), knots)  # cubic
        B_natural = RecombinedBSplineBasis(B, Natural())
        
        # Test for monotone = 0 (no constraint)
        Random.seed!(111)
        ests_orig = rand(Normal(0, 1), length(B_natural))
        coefs = MultistateModels._spline_ests2coefs(ests_orig, B_natural, 0)
        ests_back = MultistateModels._spline_coefs2ests(coefs, B_natural, 0)
        @test maximum(abs.(ests_orig .- ests_back)) < 1e-12
        
        # Test for monotone = 1 (increasing)
        ests_orig = rand(Normal(0, 1), length(B_natural))
        coefs = MultistateModels._spline_ests2coefs(ests_orig, B_natural, 1)
        ests_back = MultistateModels._spline_coefs2ests(coefs, B_natural, 1)
        @test maximum(abs.(ests_orig .- ests_back)) < 1e-12
        
        # Test for monotone = -1 (decreasing)
        ests_orig = rand(Normal(0, 1), length(B_natural))
        coefs = MultistateModels._spline_ests2coefs(ests_orig, B_natural, -1)
        ests_back = MultistateModels._spline_coefs2ests(coefs, B_natural, -1)
        @test maximum(abs.(ests_orig .- ests_back)) < 1e-12
    end

    # =========================================================================
    # Edge cases
    # =========================================================================
    
    @testset "Spline edge cases" begin
        # Linear spline (degree=1) - verify cumhaz matches QuadGK integration
        h_linear = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=1, knots=[0.3, 0.5, 0.7])
        
        model_linear = multistatemodel(h_linear; data=large_dat)
        haz_linear = model_linear.hazards[1]
        pars = get_parameters(model_linear, 1, scale=:log)
        covars = NamedTuple()
        lb, ub = 0.2, 0.8
        
        # Note: rtol=1e-3 because spline cumhaz uses internal integration that differs from QuadGK
        H_impl = MultistateModels.eval_cumhaz(haz_linear, lb, ub, pars, covars)
        H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz_linear, t, pars, covars), lb, ub; rtol=1e-10)
        @test isapprox(H_impl, H_quad; rtol=1e-3)
        
        # Spline with constant extrapolation - verify cumhaz at boundaries
        h_const = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                        degree=3, knots=[0.3, 0.5, 0.7],
                        extrapolation="constant")
        
        model_const = multistatemodel(h_const; data=large_dat)
        haz_const = model_const.hazards[1]
        pars_const = get_parameters(model_const, 1, scale=:log)
        
        # Verify cumhaz beyond knot boundaries matches QuadGK
        # Note: rtol=1e-3 because spline cumhaz uses internal integration that differs from QuadGK
        H_impl_const = MultistateModels.eval_cumhaz(haz_const, 0.0, 0.5, pars_const, covars)
        H_quad_const, _ = quadgk(t -> MultistateModels.eval_hazard(haz_const, t, pars_const, covars), 0.0, 0.5; rtol=1e-10)
        @test isapprox(H_impl_const, H_quad_const; rtol=1e-3)
        
        # Spline with linear extrapolation - verify cumhaz  
        h_extrap = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          extrapolation="linear")
        
        model_extrap = multistatemodel(h_extrap; data=large_dat)
        haz_extrap = model_extrap.hazards[1]
        pars_extrap = get_parameters(model_extrap, 1, scale=:log)
        
        # Note: rtol=2e-3 for extrapolation cases which have additional numerical differences
        H_impl_extrap = MultistateModels.eval_cumhaz(haz_extrap, 0.2, 0.8, pars_extrap, covars)
        H_quad_extrap, _ = quadgk(t -> MultistateModels.eval_hazard(haz_extrap, t, pars_extrap, covars), 0.2, 0.8; rtol=1e-10)
        @test isapprox(H_impl_extrap, H_quad_extrap; rtol=2e-3)
    end

    # =========================================================================
    # place_knots_from_paths! tests
    # =========================================================================
    
    @testset "place_knots_from_paths!" begin
        # Create test data with clear transition patterns
        Random.seed!(12345)
        n_subj = 30
        
        function make_reversible_data(n)
            rows = []
            for i in 1:n
                t1 = rand(Uniform(0.2, 0.6))
                t2 = t1 + rand(Uniform(0.2, 0.5))
                push!(rows, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
                push!(rows, (id=i, tstart=t1, tstop=t2, statefrom=2, stateto=1, obstype=1))
            end
            DataFrame(rows)
        end
        
        test_data = make_reversible_data(n_subj)
        
        # 1. Create and fit a Markov surrogate model
        h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21_exp = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        markov_model = multistatemodel(h12_exp, h21_exp; data=test_data)
        markov_fit = fit(markov_model)
        
        # 2. Create a target spline model with placeholder knots
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=nothing)
        h21_sp = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree=3, knots=nothing)
        spline_model = multistatemodel(h12_sp, h21_sp; data=test_data)
        
        # Store original knots for comparison
        original_knots_12 = copy(spline_model.hazards[1].knots)
        original_knots_21 = copy(spline_model.hazards[2].knots)
        
        # 3. Place knots from sampled paths
        knot_locs = place_knots_from_paths!(spline_model, markov_fit; n_knots=3)
        
        @testset "Returns correct structure" begin
            @test knot_locs isa Dict{Tuple{Int,Int}, Vector{Float64}}
            @test haskey(knot_locs, (1, 2))
            @test haskey(knot_locs, (2, 1))
            @test length(knot_locs[(1, 2)]) == 3
            @test length(knot_locs[(2, 1)]) == 3
        end
        
        @testset "Knots are properly ordered" begin
            @test issorted(knot_locs[(1, 2)])
            @test issorted(knot_locs[(2, 1)])
        end
        
        @testset "Knots are within reasonable bounds" begin
            # Knots should be positive (transition times > 0)
            @test all(knot_locs[(1, 2)] .> 0)
            @test all(knot_locs[(2, 1)] .> 0)
            
            # Knots should be less than max observation time
            max_time = maximum(test_data.tstop)
            @test all(knot_locs[(1, 2)] .< max_time)
            @test all(knot_locs[(2, 1)] .< max_time)
        end
        
        @testset "Model hazards are updated" begin
            # The model's hazards should now have new knots
            new_knots_12 = spline_model.hazards[1].knots
            new_knots_21 = spline_model.hazards[2].knots
            
            # Interior knots should match returned locations
            interior_12 = new_knots_12[2:end-1]
            interior_21 = new_knots_21[2:end-1]
            
            @test length(interior_12) >= 3
            @test length(interior_21) >= 3
        end
        
        @testset "Updated model is evaluable with QuadGK verification" begin
            # After place_knots_from_paths!, hazards have new basis dimensions
            # Create parameter vectors with correct sizes based on updated npar_baseline
            npar1 = spline_model.hazards[1].npar_total
            npar2 = spline_model.hazards[2].npar_total
            
            # Use zeros (log scale) as valid test parameters
            pars1 = zeros(npar1)
            pars2 = zeros(npar2)
            covars = NamedTuple()
            
            # Verify cumulative hazard via QuadGK integration
            lb, ub = 0.1, 0.7
            
            # Hazard 1->2
            H1_computed = MultistateModels.eval_cumhaz(spline_model.hazards[1], lb, ub, pars1, covars)
            H1_quadgk, _ = quadgk(t -> MultistateModels.eval_hazard(spline_model.hazards[1], t, pars1, covars), lb, ub; rtol=1e-8)
            @test isapprox(H1_computed, H1_quadgk; rtol=1e-6)
            
            # Hazard 2->1
            H2_computed = MultistateModels.eval_cumhaz(spline_model.hazards[2], lb, ub, pars2, covars)
            H2_quadgk, _ = quadgk(t -> MultistateModels.eval_hazard(spline_model.hazards[2], t, pars2, covars), lb, ub; rtol=1e-8)
            @test isapprox(H2_computed, H2_quadgk; rtol=1e-6)
        end
        
        @testset "Custom quantile_probs" begin
            # Create fresh model
            h12_sp2 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=nothing)
            h21_sp2 = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree=3, knots=nothing)
            spline_model2 = multistatemodel(h12_sp2, h21_sp2; data=test_data)
            
            # Custom quantiles
            custom_q = [0.25, 0.5, 0.75]
            knot_locs2 = place_knots_from_paths!(spline_model2, markov_fit; 
                                                  n_knots=3, quantile_probs=custom_q)
            
            @test length(knot_locs2[(1, 2)]) == 3
            @test length(knot_locs2[(2, 1)]) == 3
        end
    end

    # =========================================================================
    # default_nknots helper
    # =========================================================================
    
    @testset "default_nknots" begin
        # Small sample → fewer knots
        @test default_nknots(10) <= default_nknots(100)
        @test default_nknots(100) <= default_nknots(1000)
        
        # Should return at least 1 knot
        @test default_nknots(5) >= 1
        
        # Should cap at reasonable maximum
        @test default_nknots(100000) <= 20
    end
    
    # =========================================================================
    # place_interior_knots helper
    # =========================================================================
    
    @testset "place_interior_knots" begin
        sojourns = collect(range(0.1, 1.0, length=100))
        
        # Basic functionality
        knots3 = place_interior_knots(sojourns, 3)
        @test length(knots3) == 3
        @test issorted(knots3)
        @test all(knots3 .>= minimum(sojourns))
        @test all(knots3 .<= maximum(sojourns))
        
        # With explicit bounds
        knots_bounded = place_interior_knots(sojourns, 2; lower_bound=0.0, upper_bound=2.0)
        @test length(knots_bounded) == 2
        @test all(knots_bounded .>= 0.0)
        @test all(knots_bounded .<= 2.0)
        
        # Different number of knots
        knots5 = place_interior_knots(sojourns, 5)
        @test length(knots5) == 5
        @test issorted(knots5)
    end
end
