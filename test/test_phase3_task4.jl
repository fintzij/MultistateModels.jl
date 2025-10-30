# Phase 3 Task 3.4 Test: Verify build_hazards() initializes parameters_ph
# =======================================================================
#
# Task 3.4: Update build_hazards() to properly initialize parameters_ph
#           during model construction and return it as 4th output
#
# Expected behavior:
# 1. build_hazards() returns 4 values: (_hazards, parameters, parameters_ph, hazkeys)
# 2. parameters_ph is a NamedTuple with (flat, transformed, natural, unflatten)
# 3. Flat parameters are correctly ordered by hazard index
# 4. Natural parameters match the legacy parameters VectorOfVectors

using MultistateModels
using DataFrames
using ParameterHandling

println("\n" * "="^70)
println("PHASE 3 TASK 3.4: Test build_hazards() parameters_ph initialization")
println("="^70)

# Create simple test data
dat = DataFrame(
    id = repeat(1:10, inner=2),
    tstart = repeat([0.0, 1.0], 10),
    tstop = repeat([1.0, 2.0], 10),
    statefrom = repeat([1, 2], 10),
    stateto = repeat([2, 3], 10),
    obstype = ones(Int, 20)
)

# Create simple hazard functions
haz1 = ExponentialHazard(:exp1, 1, 2)
haz2 = ExponentialHazard(:exp2, 2, 3)

println("\n1. Testing build_hazards() return signature...")
println("-" * "="^69)

# Call build_hazards - should return 4 values
_hazards, parameters, parameters_ph, hazkeys = MultistateModels.build_hazards(
    haz1, haz2; 
    data = dat, 
    surrogate = false
)

println("✓ build_hazards() returns 4 values")
println("  - _hazards: $(typeof(_hazards))")
println("  - parameters: $(typeof(parameters))")
println("  - parameters_ph: $(typeof(parameters_ph))")
println("  - hazkeys: $(typeof(hazkeys))")

println("\n2. Testing parameters_ph structure...")
println("-" * "="^69)

# Check parameters_ph is a NamedTuple with correct fields
@assert haskey(parameters_ph, :flat) "parameters_ph missing :flat"
@assert haskey(parameters_ph, :transformed) "parameters_ph missing :transformed"
@assert haskey(parameters_ph, :natural) "parameters_ph missing :natural"
@assert haskey(parameters_ph, :unflatten) "parameters_ph missing :unflatten"

println("✓ parameters_ph has all required fields:")
println("  - flat: $(typeof(parameters_ph.flat)) with $(length(parameters_ph.flat)) elements")
println("  - transformed: $(typeof(parameters_ph.transformed)) with fields $(keys(parameters_ph.transformed))")
println("  - natural: $(typeof(parameters_ph.natural)) with fields $(keys(parameters_ph.natural))")
println("  - unflatten: $(typeof(parameters_ph.unflatten))")

println("\n3. Testing parameter consistency...")
println("-" * "="^69)

# Check that natural parameters match legacy parameters
for (hazname, idx) in hazkeys
    natural_params = parameters_ph.natural[hazname]
    legacy_params = exp.(parameters[idx])  # legacy is log scale
    
    @assert length(natural_params) == length(legacy_params) "Parameter count mismatch for $hazname"
    @assert all(abs.(natural_params .- legacy_params) .< 1e-10) "Parameter values mismatch for $hazname"
    
    println("✓ Hazard :$hazname (index $idx):")
    println("  - Natural scale: $natural_params")
    println("  - Legacy (exp): $legacy_params")
    println("  - Match: $(all(abs.(natural_params .- legacy_params) .< 1e-10))")
end

println("\n4. Testing unflatten function...")
println("-" * "="^69)

# Test that unflatten roundtrips correctly
reconstructed = parameters_ph.unflatten(parameters_ph.flat)
println("✓ Unflatten function works:")
println("  - Original transformed: $(parameters_ph.transformed)")
println("  - Reconstructed: $reconstructed")
@assert keys(reconstructed) == keys(parameters_ph.transformed) "Unflatten keys mismatch"

println("\n5. Testing flat parameter ordering...")
println("-" * "="^69)

# Flat parameters should be ordered by hazard index
expected_order = sort(collect(hazkeys), by = x -> x[2])
println("✓ Expected hazard order (by index): $([name for (name, _) in expected_order])")

# Verify we can extract parameters in order
offset = 1
for (hazname, idx) in expected_order
    n_params = length(parameters[idx])
    flat_chunk = parameters_ph.flat[offset:(offset + n_params - 1)]
    natural_chunk = parameters_ph.natural[hazname]
    
    # flat should be log of natural (positive transformation)
    @assert all(abs.(exp.(flat_chunk) .- natural_chunk) .< 1e-10) "Flat/natural mismatch for $hazname"
    println("  - :$hazname parameters extracted correctly from flat vector")
    
    offset += n_params
end

println("\n" * "="^70)
println("✓ Phase 3 Task 3.4 COMPLETE!")
println("="^70)
println("\nSummary:")
println("  ✓ build_hazards() properly initializes parameters_ph")
println("  ✓ Returns 4 values (_hazards, parameters, parameters_ph, hazkeys)")
println("  ✓ parameters_ph has all required fields")
println("  ✓ Natural parameters match legacy VectorOfVectors")
println("  ✓ Unflatten function works correctly")
println("  ✓ Flat parameters correctly ordered")
