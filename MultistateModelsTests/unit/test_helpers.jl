# =============================================================================
# Helper Utility Tests
# =============================================================================
#
# Tests that verify critical algorithmic correctness:
# 1. ForwardDiff compatibility (gradients/Hessians work correctly)
# 2. Batched vs sequential likelihood parity (optimization bugs)
using .TestFixtures
using ForwardDiff

# --- ForwardDiff compatibility -------------------------------------------------
# Critical: If gradients/Hessians are wrong, optimization silently fails
@testset "ForwardDiff compatibility" begin
    using MultistateModels: ExactData, loglik_exact
    
    @testset "gradient computation" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [5.0, 7.0],
            statefrom = [1, 1],
            stateto = [2, 2],
            obstype = [1, 1],
            age = [30.0, 50.0]
        )
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.1), log(1.0), 0.01],))
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = model.parameters.flat
        
        grad = ForwardDiff.gradient(p -> loglik_exact(p, exact_data; neg=false), pars)
        @test length(grad) == length(pars)
        @test all(isfinite.(grad))
    end
    
    @testset "Hessian computation" begin
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [4.0, 6.0, 3.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1],
            trt = [0.0, 1.0, 0.0]
        )
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [log(0.2), 0.3],))
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = model.parameters.flat
        
        hess = ForwardDiff.hessian(p -> loglik_exact(p, exact_data; neg=false), pars)
        @test size(hess) == (length(pars), length(pars))
        @test all(isfinite.(hess))
        @test issymmetric(hess)
    end
end

# --- Batched vs sequential parity ----------------------------------------------
# Critical: Batched optimization must give same answer as sequential
@testset "batched_vs_sequential_parity" begin
    using MultistateModels: SMPanelData, loglik_semi_markov!, loglik_semi_markov_batched!
    
    # Illness-death model tests batched path likelihood
    dat = DataFrame(
        id = [1, 1, 2, 2, 3],
        tstart = [0.0, 3.0, 0.0, 2.0, 0.0],
        tstop = [3.0, 7.0, 2.0, 5.0, 6.0],
        statefrom = [1, 2, 1, 1, 1],
        stateto = [2, 3, 1, 3, 3],
        obstype = [1, 1, 1, 1, 1],
        age = [40.0, 40.0, 50.0, 50.0, 60.0]
    )
    h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    h23 = Hazard(@formula(0 ~ 1 + age), "gom", 2, 3)
    model = multistatemodel(h12, h13, h23; data = dat)
    set_parameters!(model, (
        h12 = [log(0.1), log(1.2), 0.01],
        h13 = [log(0.05)],
        h23 = [log(0.15), 0.02, 0.01]
    ))
    
    base_paths = MultistateModels.extract_paths(model)
    n_subjects = length(base_paths)
    n_paths = 3
    nested_paths = [[deepcopy(base_paths[i]) for _ in 1:n_paths] for i in 1:n_subjects]
    weights = [ones(n_paths) for _ in 1:n_subjects]
    smpanel = SMPanelData(model, nested_paths, weights)
    pars = model.parameters.flat
    
    logliks_seq = [zeros(n_paths) for _ in 1:n_subjects]
    logliks_bat = [zeros(n_paths) for _ in 1:n_subjects]
    loglik_semi_markov!(pars, logliks_seq, smpanel)
    loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
    
    for i in 1:n_subjects, j in 1:n_paths
        @test isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
    end
end

# --- Phase 1 Parameter Handling Tests -----------------------------------------
# Tests for NamedTuple parameter structure with named fields

@testset "build_hazard_params - NamedTuple structure" begin
    using MultistateModels: build_hazard_params
    
    @testset "Weibull baseline only" begin
        params = build_hazard_params(
            [log(1.5), log(0.2)],
            [:h12_shape, :h12_scale],
            2,
            2  # npar_total = npar_baseline
        )
        
        @test params isa NamedTuple
        @test haskey(params, :baseline)
        @test params.baseline isa NamedTuple
        @test haskey(params.baseline, :h12_shape)
        @test haskey(params.baseline, :h12_scale)
        @test params.baseline.h12_shape ≈ log(1.5)
        @test params.baseline.h12_scale ≈ log(0.2)
        @test !haskey(params, :covariates)
    end
    
    @testset "Weibull with covariates" begin
        params = build_hazard_params(
            [log(1.5), log(0.2), 0.3, 0.1],
            [:h12_shape, :h12_scale, :h12_age, :h12_sex],
            2,
            4  # npar_total
        )
        
        @test haskey(params, :baseline)
        @test haskey(params, :covariates)
        @test params.baseline.h12_shape ≈ log(1.5)
        @test params.baseline.h12_scale ≈ log(0.2)
        @test haskey(params.covariates, :h12_age)
        @test haskey(params.covariates, :h12_sex)
        @test params.covariates.h12_age ≈ 0.3
        @test params.covariates.h12_sex ≈ 0.1
    end
    
    @testset "Exponential baseline only" begin
        params = build_hazard_params(
            [log(0.5)],
            [:h13_intercept],
            1,
            1  # npar_total
        )
        
        @test params.baseline.h13_intercept ≈ log(0.5)
        @test !haskey(params, :covariates)
    end
    
    @testset "Error on mismatched lengths" begin
        @test_throws AssertionError build_hazard_params(
            [1.0, 2.0],
            [:h12_shape],  # Wrong length!
            2,
            2
        )
    end
    
    @testset "Error on invalid npar_baseline" begin
        @test_throws AssertionError build_hazard_params(
            [1.0, 2.0],
            [:h12_shape, :h12_scale],
            3,  # More baseline params than total!
            2
        )
    end
end

@testset "Parameter extraction helpers" begin
    using MultistateModels: extract_baseline_values, extract_covariate_values, 
                            extract_params_vector, extract_natural_vector
    
    params_with_covars = (
        baseline = (h12_shape = log(1.5), h12_scale = log(0.2)),
        covariates = (h12_age = 0.3, h12_sex = 0.1)
    )
    
    params_no_covars = (
        baseline = (h13_intercept = log(0.8),),
    )
    
    @testset "extract_baseline_values" begin
        baseline_vals = extract_baseline_values(params_with_covars)
        @test baseline_vals ≈ [log(1.5), log(0.2)]
        @test baseline_vals isa Vector{Float64}
        
        baseline_vals_single = extract_baseline_values(params_no_covars)
        @test baseline_vals_single ≈ [log(0.8)]
    end
    
    @testset "extract_covariate_values" begin
        covar_vals = extract_covariate_values(params_with_covars)
        @test covar_vals ≈ [0.3, 0.1]
        @test covar_vals isa Vector{Float64}
        
        covar_vals_empty = extract_covariate_values(params_no_covars)
        @test isempty(covar_vals_empty)
        @test covar_vals_empty isa Vector{Float64}
    end
    
    @testset "extract_params_vector" begin
        all_params = extract_params_vector(params_with_covars)
        @test all_params ≈ [log(1.5), log(0.2), 0.3, 0.1]
        
        baseline_only = extract_params_vector(params_no_covars)
        @test baseline_only ≈ [log(0.8)]
    end
    
    @testset "extract_natural_vector" begin
        natural_vals = extract_natural_vector(params_with_covars)
        @test natural_vals ≈ [1.5, 0.2, 0.3, 0.1]
        
        natural_baseline = extract_natural_vector(params_no_covars)
        @test natural_baseline ≈ [0.8]
    end
end

@testset "ParameterHandling.jl with nested NamedTuples" begin
    using ParameterHandling
    using MultistateModels: build_hazard_params
    
    @testset "Flatten and unflatten with named fields" begin
        # Build parameter structure with named NamedTuples
        params = (
            h12 = build_hazard_params(
                [log(1.5), log(0.2), 0.3, 0.1],
                [:h12_shape, :h12_scale, :h12_age, :h12_sex],
                2,
                4  # npar_total
            ),
            h23 = build_hazard_params(
                [log(0.8)],
                [:h23_intercept],
                1,
                1  # npar_total
            )
        )
        
        # Flatten and unflatten
        flat, unflatten = ParameterHandling.flatten(params)
        reconstructed = unflatten(flat)
        
        # Verify structure preserved
        @test reconstructed.h12.baseline.h12_shape ≈ log(1.5)
        @test reconstructed.h12.baseline.h12_scale ≈ log(0.2)
        @test reconstructed.h12.covariates.h12_age ≈ 0.3
        @test reconstructed.h12.covariates.h12_sex ≈ 0.1
        @test reconstructed.h23.baseline.h23_intercept ≈ log(0.8)
        
        # Test modification (as in optimization)
        modified_flat = flat .+ 0.1
        modified = unflatten(modified_flat)
        @test modified.h12.baseline.h12_shape ≈ log(1.5) + 0.1
        @test modified.h12.baseline.h12_scale ≈ log(0.2) + 0.1
        @test modified.h12.covariates.h12_age ≈ 0.3 + 0.1
        @test modified.h23.baseline.h23_intercept ≈ log(0.8) + 0.1
    end
    
    @testset "Named access works correctly" begin
        params = (
            h12 = (
                baseline = (h12_shape = log(2.0), h12_scale = log(1.0)),
                covariates = (h12_trt = 0.5,)
            ),
        )
        
        # Test that we can access by name
        @test params.h12.baseline.h12_shape == log(2.0)
        @test params.h12.baseline.h12_scale == log(1.0)
        @test params.h12.covariates.h12_trt == 0.5
        
        # Flatten/unflatten preserves named access
        flat, unflatten = ParameterHandling.flatten(params)
        restored = unflatten(flat)
        @test restored.h12.baseline.h12_shape == log(2.0)
        @test restored.h12.covariates.h12_trt == 0.5
    end
end
