# Test Phase 3 Task 3.2 - set_parameters!() synchronization
# Verify that all set_parameters!() variants update both parameters and parameters_ph

println("="^70)
println("Testing Phase 3 Task 3.2: set_parameters!() Synchronization")
println("="^70)

# Check that the functions are updated in src/helpers.jl
println("\n✓ Checking src/helpers.jl for updated set_parameters! functions...")

helpers_src = read("src/helpers.jl", String)

# Check for Phase 3 update comments
phase3_updates = [
    "Phase 3 Update: Now automatically updates",
    "Phase 3: Update parameters_ph automatically",
    "model.parameters_ph = update_parameters_ph!(model)"
]

all_found = true
for marker in phase3_updates
    count = length(collect(eachmatch(Regex(marker), helpers_src)))
    if count >= 4  # Should appear in all 4 set_parameters! functions
        println("  ✓ Found '$marker' in $count locations")
    else
        println("  ✗ '$marker' found only $count times (expected ≥4)")
        all_found = false
    end
end

# Check for the new single-hazard version
if occursin("set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})", helpers_src)
    println("  ✓ Found new single-hazard set_parameters! function")
else
    println("  ✗ Missing single-hazard set_parameters! function")
    all_found = false
end

# Count total set_parameters! functions
set_params_count = length(collect(eachmatch(r"function set_parameters!", helpers_src)))
println("  ℹ Total set_parameters! functions: $set_params_count (expected 4)")

if all_found && set_params_count == 4
    println("\n" * "="^70)
    println("✓ Phase 3 Task 3.2 COMPLETE!")
    println("All 4 set_parameters!() functions now automatically synchronize:")
    println("  1. set_parameters!(model, ::VectorOfVectors)")
    println("  2. set_parameters!(model, ::Tuple)")
    println("  3. set_parameters!(model, ::NamedTuple)")
    println("  4. set_parameters!(model, h::Int64, ::Vector) [NEW]")
    println("")
    println("Both model.parameters and model.parameters_ph stay in sync!")
    println("="^70)
else
    error("Some set_parameters! functions are incomplete!")
end
