# Test suite for ReConstructor: AD-compatible parameter flattening/unflattening
# Based on manual implementation of ModelWrappers.jl pattern

using Test
using MultistateModels
using ForwardDiff

@testset "ReConstructor Implementation" begin
    
    @testset "Type Definitions" begin
        # Test type hierarchy
        @test FlattenContinuous <: MultistateModels.FlattenTypes
        @test FlattenAll <: MultistateModels.FlattenTypes
        @test UnflattenStrict <: MultistateModels.UnflattenTypes
        @test UnflattenFlexible <: MultistateModels.UnflattenTypes
        
        # Test FlattenDefault constructors
        default1 = MultistateModels.FlattenDefault()
        @test default1.flattentype isa FlattenContinuous
        @test default1.unflattentype isa UnflattenStrict
        
        default2 = MultistateModels.FlattenDefault(UnflattenFlexible())
        @test default2.flattentype isa FlattenContinuous
        @test default2.unflattentype isa UnflattenFlexible
    end
    
    @testset "Real Number Construction" begin
        # Strict mode
        flatten_fn, unflatten_fn = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), 1.5
        )
        
        flat = flatten_fn(2.3)
        @test flat == [2.3]
        @test length(flat) == 1
        
        reconstructed = unflatten_fn(flat)
        @test reconstructed ≈ 2.3
        @test reconstructed isa Float64
        
        # Flexible mode
        flatten_fn_flex, unflatten_fn_flex = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenFlexible(), 1.5
        )
        
        flat_flex = flatten_fn_flex(2.3)
        reconstructed_flex = unflatten_fn_flex(flat_flex)
        @test reconstructed_flex ≈ 2.3
    end
    
    @testset "Vector Construction" begin
        # Float64 vector - should flatten
        x_float = [1.0, 2.0, 3.0]
        
        flatten_fn, unflatten_fn = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), x_float
        )
        
        flat = flatten_fn(x_float)
        @test flat ≈ [1.0, 2.0, 3.0]
        @test length(flat) == 3
        
        reconstructed = unflatten_fn(flat)
        @test reconstructed ≈ x_float
        
        # Integer vector with FlattenContinuous - should NOT flatten
        x_int = [1, 2, 3]
        
        flatten_fn_int, unflatten_fn_int = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), x_int
        )
        
        flat_int = flatten_fn_int(x_int)
        @test isempty(flat_int)  # Should return empty vector
        
        reconstructed_int = unflatten_fn_int(flat_int)
        @test reconstructed_int == x_int  # Should return original
        
        # Integer vector with FlattenAll - should flatten
        flatten_fn_all, unflatten_fn_all = MultistateModels.construct_flatten(
            Float64, FlattenAll(), UnflattenStrict(), x_int
        )
        
        flat_all = flatten_fn_all(x_int)
        @test !isempty(flat_all)
        @test length(flat_all) == 3
    end
    
    @testset "Tuple Construction" begin
        x = (1.5, 2.3, 3.7)
        
        flatten_fn, unflatten_fn = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), x
        )
        
        flat = flatten_fn(x)
        @test flat ≈ [1.5, 2.3, 3.7]
        @test length(flat) == 3
        
        reconstructed = unflatten_fn(flat)
        @test reconstructed isa Tuple
        @test length(reconstructed) == 3
        @test all(reconstructed .≈ x)
        
        # Nested tuple
        x_nested = ((1.0, 2.0), (3.0, 4.0, 5.0))
        
        flatten_fn_nested, unflatten_fn_nested = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), x_nested
        )
        
        flat_nested = flatten_fn_nested(x_nested)
        @test flat_nested ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
        @test length(flat_nested) == 5
        
        reconstructed_nested = unflatten_fn_nested(flat_nested)
        @test reconstructed_nested isa Tuple
        @test length(reconstructed_nested) == 2
        @test reconstructed_nested[1] isa Tuple
        @test length(reconstructed_nested[1]) == 2
        @test length(reconstructed_nested[2]) == 3
    end
    
    @testset "NamedTuple Construction - Strict Mode" begin
        x = (shape = 1.5, scale = 0.2)
        
        flatten_fn, unflatten_fn = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), x
        )
        
        flat = flatten_fn(x)
        @test flat ≈ [1.5, 0.2]
        @test length(flat) == 2
        
        reconstructed = unflatten_fn(flat)
        @test reconstructed isa NamedTuple
        @test haskey(reconstructed, :shape)
        @test haskey(reconstructed, :scale)
        @test reconstructed.shape ≈ 1.5
        @test reconstructed.scale ≈ 0.2
        
        # Nested NamedTuple
        x_nested = (
            baseline = (shape = 1.5, scale = 0.2),
            covariates = (age = 0.3, sex = 0.1)
        )
        
        flatten_fn_nested, unflatten_fn_nested = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenStrict(), x_nested
        )
        
        flat_nested = flatten_fn_nested(x_nested)
        @test flat_nested ≈ [1.5, 0.2, 0.3, 0.1]
        @test length(flat_nested) == 4
        
        reconstructed_nested = unflatten_fn_nested(flat_nested)
        @test reconstructed_nested isa NamedTuple
        @test reconstructed_nested.baseline.shape ≈ 1.5
        @test reconstructed_nested.baseline.scale ≈ 0.2
        @test reconstructed_nested.covariates.age ≈ 0.3
        @test reconstructed_nested.covariates.sex ≈ 0.1
    end
    
    @testset "NamedTuple Construction - Flexible Mode (AD)" begin
        x = (shape = 1.5, scale = 0.2)
        
        flatten_fn, unflatten_fn = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenFlexible(), x
        )
        
        # Test with Float64
        flat = flatten_fn(x)
        reconstructed = unflatten_fn(flat)
        @test reconstructed.shape ≈ 1.5
        @test reconstructed.scale ≈ 0.2
        
        # Test with Dual numbers (AD)
        using ForwardDiff
        flat_dual = ForwardDiff.Dual.(flat, 1.0)
        reconstructed_dual = unflatten_fn(flat_dual)
        
        @test reconstructed_dual isa NamedTuple
        @test reconstructed_dual.shape isa ForwardDiff.Dual
        @test reconstructed_dual.scale isa ForwardDiff.Dual
        @test ForwardDiff.value(reconstructed_dual.shape) ≈ 1.5
        @test ForwardDiff.value(reconstructed_dual.scale) ≈ 0.2
        
        # Nested NamedTuple with Dual
        x_nested = (
            baseline = (shape = 1.5, scale = 0.2),
            covariates = (age = 0.3, sex = 0.1)
        )
        
        flatten_fn_nested, unflatten_fn_nested = MultistateModels.construct_flatten(
            Float64, FlattenContinuous(), UnflattenFlexible(), x_nested
        )
        
        flat_nested = flatten_fn_nested(x_nested)
        flat_nested_dual = ForwardDiff.Dual.(flat_nested, 1.0)
        reconstructed_nested_dual = unflatten_fn_nested(flat_nested_dual)
        
        @test reconstructed_nested_dual.baseline.shape isa ForwardDiff.Dual
        @test reconstructed_nested_dual.covariates.age isa ForwardDiff.Dual
        @test ForwardDiff.value(reconstructed_nested_dual.baseline.shape) ≈ 1.5
        @test ForwardDiff.value(reconstructed_nested_dual.covariates.age) ≈ 0.3
    end
    
    @testset "ReConstructor API" begin
        params = (
            baseline = (shape = 1.5, scale = 0.2),
            covariates = (age = 0.3, sex = 0.1)
        )
        
        # Build with strict mode
        rc_strict = MultistateModels.ReConstructor(params)
        
        @test rc_strict isa MultistateModels.ReConstructor
        @test rc_strict.default.unflattentype isa UnflattenStrict
        
        # Test flatten
        flat = MultistateModels.flatten(rc_strict, params)
        @test flat ≈ [1.5, 0.2, 0.3, 0.1]
        @test flat isa Vector{Float64}
        
        # Test unflatten
        reconstructed = MultistateModels.unflatten(rc_strict, flat)
        @test reconstructed.baseline.shape ≈ 1.5
        @test reconstructed.covariates.age ≈ 0.3
        
        # Build with flexible mode
        rc_flex = MultistateModels.ReConstructor(params, unflattentype=UnflattenFlexible())
        
        @test rc_flex.default.unflattentype isa UnflattenFlexible
        
        # Test flattenAD (should be same as flatten for Float64)
        flat_ad = MultistateModels.flattenAD(rc_flex, params)
        @test flat_ad ≈ flat
        
        # Test unflattenAD with Dual numbers
        flat_dual = ForwardDiff.Dual.(flat, 1.0)
        reconstructed_dual = MultistateModels.unflattenAD(rc_flex, flat_dual)
        
        @test reconstructed_dual.baseline.shape isa ForwardDiff.Dual
        @test reconstructed_dual.baseline.scale isa ForwardDiff.Dual
        @test reconstructed_dual.covariates.age isa ForwardDiff.Dual
        @test ForwardDiff.value(reconstructed_dual.baseline.shape) ≈ 1.5
    end
    
    @testset "AD Compatibility - ForwardDiff Integration" begin
        params = (
            baseline = (shape = 1.5, scale = 0.2),
            covariates = (age = 0.3, sex = 0.1)
        )
        
        rc = MultistateModels.ReConstructor(params, unflattentype=UnflattenFlexible())
        
        # Define function that uses unflattenAD
        function test_function(flat_params)
            p = MultistateModels.unflattenAD(rc, flat_params)
            return p.baseline.shape^2 + p.baseline.scale * p.covariates.age
        end
        
        flat = MultistateModels.flatten(rc, params)
        
        # Test that ForwardDiff works
        grad = ForwardDiff.gradient(test_function, flat)
        
        @test length(grad) == 4
        @test !any(isnan.(grad))
        
        # Manual verification of gradient
        # f = shape^2 + scale * age
        # ∂f/∂shape = 2*shape = 2*1.5 = 3.0
        # ∂f/∂scale = age = 0.3
        # ∂f/∂age = scale = 0.2
        # ∂f/∂sex = 0
        @test grad[1] ≈ 3.0  # ∂f/∂shape
        @test grad[2] ≈ 0.3  # ∂f/∂scale
        @test grad[3] ≈ 0.2  # ∂f/∂age
        @test grad[4] ≈ 0.0  # ∂f/∂sex
    end
    
    @testset "Performance - Zero Allocation" begin
        params = (
            baseline = (shape = 1.5, scale = 0.2),
            covariates = (age = 0.3, sex = 0.1)
        )
        
        rc = MultistateModels.ReConstructor(params, unflattentype=UnflattenFlexible())
        flat = MultistateModels.flatten(rc, params)
        
        # Warm-up
        MultistateModels.unflatten(rc, flat)
        MultistateModels.flatten(rc, params)
        
        # Measure allocations (should be minimal)
        allocs_unflatten = @allocated MultistateModels.unflatten(rc, flat)
        allocs_flatten = @allocated MultistateModels.flatten(rc, params)
        
        # Some allocations acceptable (NamedTuple construction), but should be < 600 bytes
        @test allocs_unflatten < 600
        @test allocs_flatten < 600
    end
    
    @testset "Edge Cases" begin
        # Empty NamedTuple
        empty_params = NamedTuple()
        rc_empty = MultistateModels.ReConstructor(empty_params)
        flat_empty = MultistateModels.flatten(rc_empty, empty_params)
        @test isempty(flat_empty)
        
        # Single parameter
        single_param = (intercept = 0.5,)
        rc_single = MultistateModels.ReConstructor(single_param)
        flat_single = MultistateModels.flatten(rc_single, single_param)
        @test flat_single ≈ [0.5]
        reconstructed_single = MultistateModels.unflatten(rc_single, flat_single)
        @test reconstructed_single.intercept ≈ 0.5
        
        # Mixed integer and float (FlattenContinuous should skip integers)
        mixed_tuple = (1.5, [1, 2, 3], 2.3)
        rc_mixed = MultistateModels.ReConstructor(mixed_tuple)
        flat_mixed = MultistateModels.flatten(rc_mixed, mixed_tuple)
        # Should flatten both floats AND skip the integer vector (empty contribution)
        # Result: [1.5] + [] + [2.3] = [1.5, 2.3]
        @test length(flat_mixed) == 2  # Only the two floats
        @test flat_mixed[1] ≈ 1.5
        @test flat_mixed[2] ≈ 2.3
    end
end
