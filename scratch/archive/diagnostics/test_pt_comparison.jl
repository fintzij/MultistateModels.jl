# Weibull panel tests with Markov vs PhaseType proposals
using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, PhaseTypeProposal, @formula

const RNG_SEED = 0xABCD1234
const N_SUBJECTS = 1000
const MAX_TIME = 15.0
const MCEM_TOL = 0.05
const MAX_ITER = 30
const PARAM_TOL_REL = 0.35
const PROPOSAL_COMPARISON_TOL = 0.30

# Include helpers from long tests
include(joinpath(@__DIR__, "..", "MultistateModelsTests", "longtests", "longtest_config.jl"))
include(joinpath(@__DIR__, "..", "MultistateModelsTests", "longtests", "longtest_helpers.jl"))

# Data generation function (from longtest_mcem.jl)
function generate_panel_data_progressive(hazards, true_params; 
    n_subj::Int = N_SUBJECTS,
    obs_times::Vector{Float64} = collect(0.0:2.0:MAX_TIME),
    covariate_data::Union{Nothing, DataFrame} = nothing)
    
    nobs = length(obs_times) - 1
    
    template = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat(obs_times[1:end-1], n_subj),
        tstop = repeat(obs_times[2:end], n_subj),
        statefrom = ones(Int, n_subj * nobs),
        stateto = ones(Int, n_subj * nobs),
        obstype = fill(2, n_subj * nobs)
    )
    
    if !isnothing(covariate_data)
        cov_expanded = DataFrame()
        for col in names(covariate_data)
            cov_expanded[!, col] = repeat(covariate_data[!, col], inner=nobs)
        end
        template = hcat(template, cov_expanded)
    end
    
    model = multistatemodel(hazards...; data=template, initialize=false)
    
    for (haz_idx, haz_name) in enumerate(keys(true_params))
        set_parameters!(model, haz_idx, true_params[haz_name])
    end
    
    obstype_map = Dict(1 => 2, 2 => 1)
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    return sim_result[1, 1]
end

function print_param_comparison(name, true_p, est_p; pnames=nothing)
    n = length(true_p)
    if isnothing(pnames); pnames = ["p[$i]" for i in 1:n]; end
    println("\n    Parameter Comparison: $name")
    println("    ", "-"^70)
    println("    ", rpad("Parameter", 18), rpad("True", 12), rpad("Estimated", 12), rpad("Abs Diff", 12), "Rel Diff (%)")
    println("    ", "-"^70)
    for i in 1:n
        ad = abs(est_p[i] - true_p[i])
        rd = abs(true_p[i]) > 1e-10 ? 100.0 * ad / abs(true_p[i]) : NaN
        println("    ", rpad(pnames[i], 18), rpad(@sprintf("%.4f", true_p[i]), 12),
            rpad(@sprintf("%.4f", est_p[i]), 12), rpad(@sprintf("%.4f", ad), 12),
            isnan(rd) ? "N/A" : @sprintf("%.1f%%", rd))
    end
    println("    ", "-"^70)
    flush(stdout)
end

println("="^80)
println("  MCEM Weibull Panel Tests - Markov vs PhaseType Proposals")
println("="^80)
flush(stdout)

@testset "MCEM Weibull - No Covariates" begin
    println("\n  ▸ MCEM Weibull - No Covariates"); flush(stdout)
    Random.seed!(RNG_SEED + 10)
    
    # True parameters
    true_shape_12, true_scale_12 = 1.3, 0.15
    true_shape_23, true_scale_23 = 1.1, 0.12
    true_params = (h12 = [true_shape_12, true_scale_12], h23 = [true_shape_23, true_scale_23])
    
    # Generate panel data
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    panel_data = generate_panel_data_progressive((h12, h23), true_params)
    println("    Generated: $(nrow(panel_data)) rows, $(length(unique(panel_data.id))) subjects")
    flush(stdout)
    
    # =====================================================================
    # Fit with Markov proposal
    # =====================================================================
    println("\n    ▸ Fitting with Markov proposal..."); flush(stdout)
    h12_m = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23_m = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    model_m = multistatemodel(h12_m, h23_m; data=panel_data, surrogate=:markov)
    
    t_m = @elapsed fitted_m = fit(model_m; verbose=true, maxiter=MAX_ITER, tol=MCEM_TOL,
        ess_target_initial=250, max_ess=1000, compute_vcov=true, compute_ij_vcov=false, proposal=:markov)
    
    p_m = get_parameters(fitted_m; scale=:natural)
    print_param_comparison("Weibull (Markov)", 
        [true_shape_12, true_scale_12, true_shape_23, true_scale_23],
        [p_m.h12[1], p_m.h12[2], p_m.h23[1], p_m.h23[2]], 
        pnames=["shape_12", "scale_12", "shape_23", "scale_23"])
    println("    Time: $(round(t_m, digits=1)) seconds")
    
    # =====================================================================
    # Fit with PhaseType proposal
    # =====================================================================
    println("\n    ▸ Fitting with PhaseType proposal..."); flush(stdout)
    h12_pt = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23_pt = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    model_pt = multistatemodel(h12_pt, h23_pt; data=panel_data, surrogate=:markov)
    
    t_pt = @elapsed fitted_pt = fit(model_pt; verbose=true, maxiter=MAX_ITER, tol=MCEM_TOL,
        ess_target_initial=250, max_ess=1000, compute_vcov=true, compute_ij_vcov=false, 
        proposal=PhaseTypeProposal(n_phases=3))
    
    p_pt = get_parameters(fitted_pt; scale=:natural)
    print_param_comparison("Weibull (PhaseType)", 
        [true_shape_12, true_scale_12, true_shape_23, true_scale_23],
        [p_pt.h12[1], p_pt.h12[2], p_pt.h23[1], p_pt.h23[2]], 
        pnames=["shape_12", "scale_12", "shape_23", "scale_23"])
    println("    Time: $(round(t_pt, digits=1)) seconds")
    
    # =====================================================================
    # Compare proposals
    # =====================================================================
    println("\n" * "="^60)
    println("  Markov vs PhaseType Comparison:")
    println("="^60)
    params_m = [p_m.h12[1], p_m.h12[2], p_m.h23[1], p_m.h23[2]]
    params_pt = [p_pt.h12[1], p_pt.h12[2], p_pt.h23[1], p_pt.h23[2]]
    pnames = ["shape_12", "scale_12", "shape_23", "scale_23"]
    println("    ", rpad("Parameter", 15), rpad("Markov", 12), rpad("PhaseType", 12), "Rel Diff (%)")
    for (i, name) in enumerate(pnames)
        rd = 100.0 * abs(params_pt[i] - params_m[i]) / abs(params_m[i])
        println("    ", rpad(name, 15), rpad(@sprintf("%.4f", params_m[i]), 12), 
                rpad(@sprintf("%.4f", params_pt[i]), 12), @sprintf("%.1f%%", rd))
    end
    
    pk_m = fitted_m.ConvergenceRecords.psis_pareto_k
    pk_pt = fitted_pt.ConvergenceRecords.psis_pareto_k
    println("\n  Pareto-k (lower is better):")
    println("    Markov:    median=", @sprintf("%.3f", median(filter(!isnan, pk_m))), 
            ", max=", @sprintf("%.3f", maximum(filter(!isnan, pk_m))))
    println("    PhaseType: median=", @sprintf("%.3f", median(filter(!isnan, pk_pt))), 
            ", max=", @sprintf("%.3f", maximum(filter(!isnan, pk_pt))))
    flush(stdout)
    
    @testset "Parameter recovery (Markov)" begin
        @test isapprox(p_m.h12[1], true_shape_12; rtol=PARAM_TOL_REL)
        @test isapprox(p_m.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
        @test isapprox(p_m.h23[1], true_shape_23; rtol=PARAM_TOL_REL)
        @test isapprox(p_m.h23[2], true_scale_23; rtol=PARAM_TOL_REL)
    end
    
    @testset "Parameter recovery (PhaseType)" begin
        @test isapprox(p_pt.h12[1], true_shape_12; rtol=PARAM_TOL_REL)
        @test isapprox(p_pt.h12[2], true_scale_12; rtol=PARAM_TOL_REL)
        @test isapprox(p_pt.h23[1], true_shape_23; rtol=PARAM_TOL_REL)
        @test isapprox(p_pt.h23[2], true_scale_23; rtol=PARAM_TOL_REL)
    end
    
    @testset "Markov vs PhaseType agreement" begin
        for (i, name) in enumerate(pnames)
            rd = abs(params_pt[i] - params_m[i]) / abs(params_m[i])
            @test rd < PROPOSAL_COMPARISON_TOL
        end
    end
end

println("\n" * "="^80)
println("  TESTS COMPLETE")
println("="^80)

println("\nIDENTICAL? ", model_m.markovsurrogate.parameters.flat == model_pt.markovsurrogate.parameters.flat)
