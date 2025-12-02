# =============================================================================
# Unit tests for spline hazard functionality
# =============================================================================

using Test
using DataFrames
using Distributions
using MultistateModels
using Random

@testset "Spline Hazards" begin

    # =========================================================================
    # Test data setup
    # =========================================================================
    
    # Simple test data
    simple_dat = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.3],
        tstop = [0.5, 1.0, 0.3, 1.0],
        statefrom = [1, 2, 1, 3],
        stateto = [2, 1, 3, 1],
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
    # Basic spline hazard creation and evaluation
    # =========================================================================
    
    @testset "Basic Spline Evaluation" begin
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, knots=[0.3, 0.5, 0.7], natural_spline=true)
        h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
                     degree=3, knots=[0.3, 0.5, 0.7])
        
        model = multistatemodel(h12, h21; data=large_dat)
        
        # Set reasonable positive parameters for hazard evaluation
        # Log-scale parameters, so exp(pars) gives positive values
        for (h, haz) in enumerate(model.hazards)
            npar = haz.npar_total
            # Use small positive log-scale values
            new_pars = fill(-1.0, npar)
            set_parameters!(model, h, new_pars)
        end
        
        # Test hazard evaluation at various times
        for (i, haz) in enumerate(model.hazards)
            pars = get_parameters(model, i, scale=:log)
            
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]
                h_val = haz(t, pars, NamedTuple())
                @test isfinite(h_val)
                @test h_val >= 0
            end
            
            # Test cumulative hazard
            H_val = MultistateModels.cumulative_hazard(haz, 0.0, 1.0, pars, NamedTuple())
            @test isfinite(H_val)
            @test H_val >= 0
        end
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
        
        # Test that evaluation works with auto-placed knots
        for (i, haz) in enumerate(auto_model.hazards)
            pars = get_parameters(auto_model, i, scale=:log)
            h_val = haz(0.5, pars, NamedTuple())
            @test isfinite(h_val)
            @test h_val >= 0
        end
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
    # Time transformation support
    # =========================================================================
    
    @testset "Time Transform for Splines" begin
        h12_tt = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                        degree=3, knots=[0.3, 0.5, 0.7],
                        natural_spline=true,
                        time_transform=true)
        h21_tt = Hazard(@formula(0 ~ 1), "sp", 2, 1; 
                        degree=3, knots=[0.3, 0.5, 0.7],
                        time_transform=true)
        
        tt_model = multistatemodel(h12_tt, h21_tt; data=large_dat)
        
        for (i, haz) in enumerate(tt_model.hazards)
            pars = get_parameters(tt_model, i, scale=:log)
            
            # Create covariates
            covars = haz.has_covariates ? (x = 0.5,) : NamedTuple()
            linpred = MultistateModels._linear_predictor(pars, covars, haz)
            
            # Test _time_transform_hazard
            for t in [0.1, 0.5, 0.9]
                h_val = MultistateModels._time_transform_hazard(haz, pars, t, linpred)
                @test isfinite(h_val)
                @test h_val >= 0
            end
            
            # Test _time_transform_cumhaz
            H_val = MultistateModels._time_transform_cumhaz(haz, pars, 0.0, 1.0, linpred)
            @test isfinite(H_val)
            @test H_val >= 0
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
        
        # Update model with rectified parameters
        ptr = MultistateModels.get_elem_ptr(model.parameters)
        for h in eachindex(model.hazards)
            rectified_pars = ests_after[ptr[h]:ptr[h+1]-1]
            set_parameters!(model, h, rectified_pars)
        end
        
        # Hazards should still evaluate correctly after rectification
        for (i, haz) in enumerate(model.hazards)
            pars = get_parameters(model, i, scale=:log)
            h_val = haz(0.5, pars, NamedTuple())
            @test isfinite(h_val)
            @test h_val >= 0
        end
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
        # Linear spline (degree=1)
        h_linear = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=1, knots=[0.3, 0.5, 0.7])
        
        model_linear = multistatemodel(h_linear; data=large_dat)
        pars = get_parameters(model_linear, 1, scale=:log)
        
        # Should evaluate without error
        @test isfinite(model_linear.hazards[1](0.5, pars, NamedTuple()))
        
        # Spline with flat extrapolation
        h_flat = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                        degree=3, knots=[0.3, 0.5, 0.7],
                        extrapolation="flat")
        
        model_flat = multistatemodel(h_flat; data=large_dat)
        pars_flat = get_parameters(model_flat, 1, scale=:log)
        
        # Evaluate beyond knot boundaries
        @test isfinite(model_flat.hazards[1](0.0, pars_flat, NamedTuple()))
        @test isfinite(model_flat.hazards[1](2.0, pars_flat, NamedTuple()))
        
        # Spline with linear extrapolation  
        h_extrap = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          extrapolation="linear")
        
        model_extrap = multistatemodel(h_extrap; data=large_dat)
        pars_extrap = get_parameters(model_extrap, 1, scale=:log)
        
        @test isfinite(model_extrap.hazards[1](0.5, pars_extrap, NamedTuple()))
    end
end
