# =============================================================================
# Variance-Covariance Unit Tests
# =============================================================================
#
# Quick unit tests for variance estimation functionality:
# 1. get_vcov API works correctly with different types
# 2. JK = ((n-1)/n) * IJ algebraic relationship
# 3. Variance matrices are positive semi-definite
# 4. Warnings for missing variance matrices

using .TestFixtures
using LinearAlgebra

@testset "get_vcov API" begin
    using MultistateModels: get_vcov
    
    @testset "returns correct variance type" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = [1, 2, 3, 4, 5],
            tstart = zeros(5),
            tstop = [2.0, 3.0, 4.0, 5.0, 6.0],
            statefrom = ones(Int, 5),
            stateto = [2, 2, 2, 2, 2],
            obstype = ones(Int, 5)
        )
        model = multistatemodel(h12; data=dat)
        fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=true)
        
        # All three variance types should be available
        vcov_model = get_vcov(fitted; type=:model)
        vcov_ij = get_vcov(fitted; type=:ij)
        vcov_jk = get_vcov(fitted; type=:jk)
        
        @test !isnothing(vcov_model)
        @test !isnothing(vcov_ij)
        @test !isnothing(vcov_jk)
        
        # All should have same dimensions
        @test size(vcov_model) == size(vcov_ij) == size(vcov_jk)
        
        # All should be square with size = number of parameters
        @test size(vcov_model, 1) == size(vcov_model, 2)
    end
    
    @testset "returns nothing when variance not computed" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = zeros(3),
            tstop = [2.0, 3.0, 4.0],
            statefrom = ones(Int, 3),
            stateto = [2, 2, 2],
            obstype = ones(Int, 3)
        )
        model = multistatemodel(h12; data=dat)
        
        # Fit without IJ/JK variance
        fitted = fit(model; verbose=false, compute_vcov=true, compute_ij_vcov=false, compute_jk_vcov=false)
        
        @test !isnothing(get_vcov(fitted; type=:model))
        @test isnothing(get_vcov(fitted; type=:ij))
        @test isnothing(get_vcov(fitted; type=:jk))
    end
end

@testset "JK = ((n-1)/n) * IJ algebraic identity" begin
    using MultistateModels: get_vcov
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    n_subj = 50
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = rand(n_subj) .* 5.0 .+ 1.0,
        statefrom = ones(Int, n_subj),
        stateto = fill(2, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [log(1.2), log(0.15)],))
    
    # Simulate and fit
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    simdat = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=simdat)
    fitted = fit(model_fit; verbose=false, compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=true)
    
    vcov_ij = get_vcov(fitted; type=:ij)
    vcov_jk = get_vcov(fitted; type=:jk)
    
    # Relationship should hold exactly (algebraic identity)
    n = n_subj
    expected_jk = ((n - 1) / n) * vcov_ij
    
    @test isapprox(vcov_jk, expected_jk; atol=1e-12)
end

@testset "Variance matrices positive semi-definite" begin
    using MultistateModels: get_vcov
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    n_subj = 30
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = rand(n_subj) .* 5.0 .+ 1.0,
        statefrom = ones(Int, n_subj),
        stateto = fill(2, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [log(1.1), log(0.20)],))
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    simdat = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=simdat)
    fitted = fit(model_fit; verbose=false, compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    vcov_jk = get_vcov(fitted; type=:jk)
    
    # Check eigenvalues >= 0 (with small tolerance for numerical errors)
    @test all(eigvals(Symmetric(vcov_model)) .>= -sqrt(eps()))
    @test all(eigvals(Symmetric(vcov_ij)) .>= -sqrt(eps()))
    @test all(eigvals(Symmetric(vcov_jk)) .>= -sqrt(eps()))
    
    # Check diagonals positive
    @test all(diag(vcov_model) .> 0)
    @test all(diag(vcov_ij) .> 0)
    @test all(diag(vcov_jk) .> 0)
end

@testset "Variance with panel data (Markov)" begin
    using MultistateModels: get_vcov
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    n_subj = 50
    nobs = 3
    dat = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat([0.0, 2.0, 4.0], n_subj),
        tstop = repeat([2.0, 4.0, 6.0], n_subj),
        statefrom = repeat([1, 1, 1], n_subj),
        stateto = repeat([1, 1, 1], n_subj),
        obstype = repeat([2, 2, 2], n_subj)  # Panel data
    )
    
    model = multistatemodel(h12, h23; data=dat)
    set_parameters!(model, (h12 = [log(0.2)], h23 = [log(0.15)]))
    
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    simdat = sim_result[1]
    
    model_fit = multistatemodel(h12, h23; data=simdat)
    fitted = fit(model_fit; verbose=false, compute_vcov=true, compute_ij_vcov=true)
    
    vcov_model = get_vcov(fitted; type=:model)
    vcov_ij = get_vcov(fitted; type=:ij)
    
    @test !isnothing(vcov_model)
    @test !isnothing(vcov_ij)
    @test size(vcov_model) == (2, 2)
    @test size(vcov_ij) == (2, 2)
    
    # All diagonals should be positive
    @test all(diag(vcov_model) .> 0)
    @test all(diag(vcov_ij) .> 0)
end
