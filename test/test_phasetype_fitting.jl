# =============================================================================
# Tests for Phase-Type Model Fitting and Parameter Handling
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra

@testset "Phase-Type Model Fitting" begin
    Random.seed!(42)
    
    # Create realistic test data for a 1→2→3 progressive model
    n_subjects = 100
    
    # Generate data where subjects transition 1→2→3
    data_rows = []
    for i in 1:n_subjects
        t1 = rand() * 2  # Time in state 1
        t2 = rand() * 2  # Additional time in state 2
        
        push!(data_rows, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
        push!(data_rows, (id=i, tstart=t1, tstop=t1+t2, statefrom=2, stateto=3, obstype=1))
    end
    data = DataFrame(data_rows)
    
    @testset "Parameter Accessor Functions" begin
        # Build model with phase-type hazard
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=:unstructured)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        
        # Test get_parameters returns user-facing parameters
        params = get_parameters(model)
        @test haskey(params, :h12)  # Phase-type hazard
        @test haskey(params, :h23)  # Exponential hazard
        
        # Test get_expanded_parameters returns internal parameters
        exp_params = get_expanded_parameters(model)
        @test haskey(exp_params, :h1_prog1)  # λ₁
        @test haskey(exp_params, :h1_prog2)  # λ₂
        @test haskey(exp_params, :h12_exit1) # μ₁
        @test haskey(exp_params, :h12_exit2) # μ₂
        @test haskey(exp_params, :h12_exit3) # μ₃
        @test haskey(exp_params, :h23)       # h23 transition
        
        # Test get_parameters_flat returns flat vector
        flat = get_parameters_flat(model)
        @test flat isa Vector{Float64}
        @test length(flat) == 6  # 5 phase-type params + 1 exp param
        
        # Test get_parameters_nested
        nested = get_parameters_nested(model)
        @test nested isa NamedTuple
        
        # Test get_parameters_natural
        natural = get_parameters_natural(model)
        @test natural isa NamedTuple
        
        # Test get_unflatten_fn
        unflatten = get_unflatten_fn(model)
        @test unflatten isa Function
        
        # Test round-trip: flatten → unflatten
        restored = unflatten(flat)
        for key in keys(nested)
            @test isapprox(nested[key].baseline, restored[key].baseline; atol=1e-10)
        end
    end
    
    @testset "set_parameters! with Vector{Vector}" begin
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        initialize_parameters!(model)
        
        # Set new values using Vector{Vector} format
        # h12: [λ₁, μ₁, μ₂] (2n-1 = 3 params for n=2)
        # h23: [rate]
        new_values = [
            [log(0.5), log(0.3), log(0.4)],  # h12
            [log(0.6)]                        # h23
        ]
        
        set_parameters!(model, new_values)
        
        # Verify user-facing parameters were updated
        params = get_parameters(model)
        @test isapprox(params[:h12][1], 0.5; rtol=1e-6)
        @test isapprox(params[:h12][2], 0.3; rtol=1e-6)
        @test isapprox(params[:h12][3], 0.4; rtol=1e-6)
        @test isapprox(params[:h23][1], 0.6; rtol=1e-6)
        
        # Verify expanded parameters were also updated
        exp_params = get_expanded_parameters(model)
        @test isapprox(exp_params[:h1_prog1][1], 0.5; rtol=1e-6)
        @test isapprox(exp_params[:h12_exit1][1], 0.3; rtol=1e-6)
        @test isapprox(exp_params[:h12_exit2][1], 0.4; rtol=1e-6)
    end
    
    @testset "Initialization Respects Structure" begin
        # Test all three structures initialize with uniform rates
        for structure in [:unstructured, :allequal, :prop_to_prog]
            h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=structure)
            h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
            
            model = multistatemodel(h12, h23; data=data)
            initialize_parameters!(model)
            
            exp_params = get_expanded_parameters(model)
            
            # Get all phase-type rates
            λ1 = exp_params[:h1_prog1][1]
            λ2 = exp_params[:h1_prog2][1]
            μ1 = exp_params[:h12_exit1][1]
            μ2 = exp_params[:h12_exit2][1]
            μ3 = exp_params[:h12_exit3][1]
            
            all_rates = [λ1, λ2, μ1, μ2, μ3]
            rate_range = maximum(all_rates) - minimum(all_rates)
            
            # All rates should be equal for all built-in structures
            @test rate_range < 1e-10
        end
    end
    
    @testset "Basic Fitting" begin
        # Simple 2-phase model
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:allequal)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        
        # Fit the model
        fitted = fit(model; verbose=false, compute_vcov=false)
        
        # Check return type - now returns MultistateModelFitted
        @test fitted isa MultistateModels.MultistateModelFitted
        
        # Check it's detected as phase-type fitted
        @test is_phasetype_fitted(fitted)
        
        # Check convergence
        @test get_convergence(fitted) == true
        
        # Check loglikelihood is finite
        @test isfinite(get_loglik(fitted))
        
        # Check parameters are returned (user-facing phase-type params by default)
        params = get_parameters(fitted)
        @test haskey(params, :h12)
        @test haskey(params, :h23)
        
        # Check parameters are positive (natural scale)
        @test all(params[:h12] .> 0)
        @test all(params[:h23] .> 0)
        
        # Check expanded parameters are accessible
        exp_params = get_parameters(fitted; expanded=true)
        @test haskey(exp_params, :h1_prog1)
        @test haskey(exp_params, :h12_exit1)
    end
    
    @testset "Fitting with Variance-Covariance" begin
        # Larger dataset for stable vcov
        n_large = 200
        data_rows_large = []
        for i in 1:n_large
            t1 = rand() * 2
            t2 = rand() * 2
            push!(data_rows_large, (id=i, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1))
            push!(data_rows_large, (id=i, tstart=t1, tstop=t1+t2, statefrom=2, stateto=3, obstype=1))
        end
        data_large = DataFrame(data_rows_large)
        
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data_large)
        fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=false)
        
        # Check vcov is returned
        vcov = get_vcov(fitted)
        @test !isnothing(vcov)
        @test vcov isa Matrix{Float64}
        
        # Vcov should be symmetric
        @test issymmetric(vcov)
        
        # Diagonal elements should be positive (variances)
        @test all(diag(vcov) .>= 0)
    end
    
    @testset "Access Fitted Expanded Model" begin
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        fitted = fit(model; verbose=false, compute_vcov=false)
        
        # The fitted model IS the expanded model now (no wrapper)
        @test fitted isa MultistateModels.MultistateModelFitted
        
        # The hazards should have the internal hazard structure
        @test length(fitted.hazards) == 4  # 1 prog + 2 exit + 1 h23
        
        # Loglikelihood should be accessible
        @test isfinite(get_loglik(fitted))
        
        # Access mappings
        mappings = get_mappings(fitted)
        @test !isnothing(mappings)
        
        # Access original data
        orig_data = get_original_data(fitted)
        @test orig_data isa DataFrame
        
        # Access original tmat
        orig_tmat = get_original_tmat(fitted)
        @test orig_tmat isa Matrix{Int64}
    end
    
    @testset "Parameter Round-Trip After Fitting" begin
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        model = multistatemodel(h12, h23; data=data)
        fitted = fit(model; verbose=false, compute_vcov=false)
        
        # Get user-facing parameters
        params = get_parameters(fitted)
        
        # Get expanded parameters
        exp_params = get_parameters(fitted; expanded=true)
        
        # The h23 rate should match in both representations
        @test isapprox(params[:h23][1], exp_params[:h23][1]; rtol=1e-6)
        
        # Can also use explicit accessor
        pt_params = get_phasetype_parameters(fitted)
        @test pt_params == params
    end
end
