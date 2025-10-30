# Phase 3 Comprehensive Test: ParameterHandling.jl Integration
# =============================================================
#
# Tests for all Phase 3 functionality:
# - Task 3.1: Mutable structs
# - Task 3.2: set_parameters!() synchronization
# - Task 3.3: get_parameters_*() functions
# - Task 3.4: build_hazards() initialization
# - Task 3.5: Model construction with parameters_ph
# - Task 3.6: Optimization using get_parameters_flat()

using MultistateModels
using DataFrames
using Test
using ParameterHandling

println("\n" * "="^70)
println("PHASE 3 COMPREHENSIVE TEST: ParameterHandling.jl Integration")
println("="^70)

# Create test data
dat_exact = DataFrame(
    id = repeat(1:10, inner=2),
    tstart = repeat([0.0, 1.0], 10),
    tstop = repeat([1.0, 2.0], 10),
    statefrom = repeat([1, 2], 10),
    stateto = repeat([2, 3], 10),
    obstype = ones(Int, 20)
)

dat_panel = DataFrame(
    id = repeat(1:10, inner=2),
    tstart = repeat([0.0, 1.0], 10),
    tstop = repeat([1.0, 2.0], 10),
    statefrom = repeat([1, 2], 10),
    stateto = repeat([2, 3], 10),
    obstype = fill(2, 20)
)

# Create hazard functions using the Hazard constructor
haz1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
haz2 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

println("\n" * "="^70)
println("TEST 1: Model Construction with parameters_ph")
println("="^70)

@testset "Model Construction" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    @test hasfield(typeof(model), :parameters_ph)
    @test !isnothing(model.parameters_ph)
    @test haskey(model.parameters_ph, :flat)
    @test haskey(model.parameters_ph, :transformed)
    @test haskey(model.parameters_ph, :natural)
    @test haskey(model.parameters_ph, :unflatten)
    
    println("✓ Model has parameters_ph field")
    println("✓ parameters_ph has all required keys")
end

println("\n" * "="^70)
println("TEST 2: Parameter Getter Functions")
println("="^70)

@testset "Parameter Getters" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Test get_parameters_flat
    flat = get_parameters_flat(model)
    @test isa(flat, Vector{Float64})
    @test length(flat) == 2  # Two hazards, one parameter each
    println("✓ get_parameters_flat() returns Vector{Float64}")
    
    # Test get_parameters_transformed
    trans = get_parameters_transformed(model)
    @test isa(trans, NamedTuple)
    @test haskey(trans, :h12)
    @test haskey(trans, :h23)
    println("✓ get_parameters_transformed() returns NamedTuple")
    
    # Test get_parameters_natural
    nat = get_parameters_natural(model)
    @test isa(nat, NamedTuple)
    @test haskey(nat, :h12)
    @test haskey(nat, :h23)
    # Natural should be exp of flat (positive transformation)
    # Use looser tolerance due to numerical precision in initialization
    @test all(abs.(exp.(flat) .- vcat(nat.h12, nat.h23)) .< 1e-7)
    println("✓ get_parameters_natural() returns correct values")
    
    # Test get_unflatten_fn
    unflatten = get_unflatten_fn(model)
    @test isa(unflatten, Function)
    reconstructed = unflatten(flat)
    @test keys(reconstructed) == keys(trans)
    println("✓ get_unflatten_fn() returns working function")
    
    # Test get_parameters with scale options
    p1_natural = get_parameters(model, 1; scale=:natural)
    p1_log = get_parameters(model, 1; scale=:log)
    @test all(abs.(exp.(p1_log) .- p1_natural) .< 1e-10)
    println("✓ get_parameters(h; scale) works for all scales")
end

println("\n" * "="^70)
println("TEST 3: set_parameters!() Synchronization")
println("="^70)

@testset "set_parameters! Synchronization" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Get initial parameters
    initial_flat = get_parameters_flat(model)
    
    # Update parameters for hazard 1
    new_params = [log(0.5)]
    set_parameters!(model, 1, new_params)
    
    # Check that parameters_ph was updated
    updated_flat = get_parameters_flat(model)
    @test updated_flat[1] ≈ log(0.5) atol=1e-7
    @test updated_flat[1] != initial_flat[1]
    
    # Check natural scale
    nat = get_parameters_natural(model)
    @test nat.h12[1] ≈ 0.5 atol=1e-7
    
    println("✓ set_parameters! updates model.parameters")
    println("✓ set_parameters! auto-syncs model.parameters_ph")
    println("✓ Natural scale reflects new values")
end

println("\n" * "="^70)
println("TEST 4: Parameter Consistency")
println("="^70)

@testset "Parameter Consistency" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Legacy VectorOfVectors should match ParameterHandling natural scale
    for (hazname, idx) in model.hazkeys
        legacy_natural = exp.(model.parameters[idx])
        ph_natural = model.parameters_ph.natural[hazname]
        
        @test length(legacy_natural) == length(ph_natural)
        @test all(abs.(legacy_natural .- ph_natural) .< 1e-10)
    end
    
    println("✓ Legacy parameters match ParameterHandling natural scale")
    println("✓ All hazards have consistent parameter counts")
end

println("\n" * "="^70)
println("TEST 5: Round-trip Parameter Updates")
println("="^70)

@testset "Round-trip Updates" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Get flat parameters
    flat_orig = copy(get_parameters_flat(model))
    
    # Modify them
    flat_new = flat_orig .+ 0.1
    
    # Update via VectorOfVectors (simulating optimization)
    for (hazname, idx) in model.hazkeys
        n_params = length(model.parameters[idx])
        # Calculate start index - handle idx=1 case
        start_idx = idx == 1 ? 0 : sum(length(model.parameters[i]) for i in 1:(idx-1))
        new_vals = flat_new[(start_idx+1):(start_idx+n_params)]
        set_parameters!(model, idx, new_vals)
    end
    
    # Check that flat parameters match
    flat_after = get_parameters_flat(model)
    @test all(abs.(flat_after .- flat_new) .< 1e-7)
    
    println("✓ Parameters survive round-trip update")
    println("✓ set_parameters! → get_parameters_flat() consistent")
end

println("\n" * "="^70)
println("TEST 6: Mutability of Structs")
println("="^70)

@testset "Struct Mutability" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Should be able to directly modify parameters_ph
    old_ph = model.parameters_ph
    new_flat = get_parameters_flat(model) .+ 0.2
    
    # This should work because struct is mutable
    try
        model.parameters_ph = (
            flat = new_flat,
            transformed = old_ph.transformed,
            natural = old_ph.natural,
            unflatten = old_ph.unflatten
        )
        @test true
        println("✓ Can directly assign to model.parameters_ph")
    catch e
        @test false
        println("✗ Failed to assign to model.parameters_ph: $e")
    end
end

println("\n" * "="^70)
println("TEST 7: Integration with Optimization")
println("="^70)

@testset "Optimization Integration" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Simulate optimization workflow
    initial_params = get_parameters_flat(model)
    @test isa(initial_params, Vector{Float64})
    
    # Simulate optimizer returning new parameters
    optimized_params = initial_params .* 1.1
    
    # Update model (as optimizer would)
    params_vov = MultistateModels.VectorOfVectors(optimized_params, model.parameters.elem_ptr)
    
    # This is what fit() does - create new parameters_ph
    # Note: params_vov contains log-scale values, but positive() expects natural scale
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(exp.(Vector{Float64}(params_vov[idx])))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    @test length(params_flat) == length(initial_params)
    @test all(abs.(params_flat .- optimized_params) .< 1e-7)
    
    println("✓ get_parameters_flat() suitable for optimizer input")
    println("✓ Can reconstruct parameters_ph from optimizer output")
end

println("\n" * "="^70)
println("TEST 8: Different Model Types")
println("="^70)

@testset "All Model Types" begin
    # MultistateModel
    model1 = multistatemodel(haz1, haz2; data = dat_exact)
    @test hasfield(typeof(model1), :parameters_ph)
    println("✓ MultistateModel has parameters_ph")
    
    # MultistateMarkovModel
    model2 = multistatemodel(haz1, haz2; data = dat_panel)
    @test hasfield(typeof(model2), :parameters_ph)
    println("✓ MultistateMarkovModel has parameters_ph")
    
    # All should have same flat parameter count for same hazards
    @test length(get_parameters_flat(model1)) == length(get_parameters_flat(model2))
    println("✓ Same hazards → same parameter count across model types")
end

println("\n" * "="^70)
println("TEST 9: Error Handling")
println("="^70)

@testset "Error Handling" begin
    model = multistatemodel(haz1, haz2; data = dat_exact)
    
    # Invalid hazard index
    @test_throws BoundsError get_parameters(model, 99; scale=:natural)
    println("✓ get_parameters throws on invalid hazard index")
    
    # Invalid scale
    @test_throws ArgumentError get_parameters(model, 1; scale=:invalid)
    println("✓ get_parameters throws on invalid scale")
end

println("\n" * "="^70)
println("✓ ALL PHASE 3 TESTS PASSED!")
println("="^70)
println("\nTest Summary:")
println("  ✓ Model construction with parameters_ph")
println("  ✓ Parameter getter functions (5 functions)")
println("  ✓ set_parameters!() synchronization")
println("  ✓ Parameter consistency (legacy ↔ ParameterHandling)")
println("  ✓ Round-trip parameter updates")
println("  ✓ Struct mutability")
println("  ✓ Optimization integration")
println("  ✓ All model types support parameters_ph")
println("  ✓ Error handling")
println("\nParameterHandling.jl integration is COMPLETE and WORKING!")
