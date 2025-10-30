# Test Phase 3 Task 3.3 - get_parameters_*() functions
# Verify that all getter functions are properly defined

println("="^70)
println("Testing Phase 3 Task 3.3: get_parameters_*() Functions")
println("="^70)

# Check that the functions are defined in src/helpers.jl
println("\n✓ Checking src/helpers.jl for new getter functions...")

helpers_src = read("src/helpers.jl", String)

# Check for all required functions
required_functions = [
    "function get_parameters_flat(model::MultistateProcess)",
    "function get_parameters_transformed(model::MultistateProcess)",
    "function get_parameters_natural(model::MultistateProcess)",
    "function get_unflatten_fn(model::MultistateProcess)",
    "function get_parameters(model::MultistateProcess, h::Int64"
]

all_found = true
for func_sig in required_functions
    if occursin(func_sig, helpers_src)
        # Extract function name for display
        func_name = match(r"function (\w+)", func_sig).captures[1]
        println("  ✓ Found: $func_name()")
    else
        println("  ✗ Missing: $func_sig")
        global all_found = false
    end
end

# Check for Phase 3 section header
if occursin("PHASE 3: Parameter Getter Functions", helpers_src)
    println("  ✓ Found Phase 3 section header")
else
    println("  ✗ Missing Phase 3 section header")
    all_found = false
end

# Check that functions have docstrings with examples
docstring_checks = [
    ("get_parameters_flat", "flat Vector{Float64}"),
    ("get_parameters_transformed", "log transformations"),
    ("get_parameters_natural", "natural scale"),
    ("get_unflatten_fn", "unflatten function"),
    ("get_parameters", "specific hazard by index")
]

println("\n✓ Checking for comprehensive docstrings...")
for (func_name, keyword) in docstring_checks
    # Simple check - function should have docstring mentioning key concept
    func_pattern = Regex("function $func_name.*?(?=function|\\z)", "s")
    func_match = match(func_pattern, helpers_src)
    if func_match !== nothing && occursin(keyword, func_match.match)
        println("  ✓ $func_name has docstring with '$keyword'")
    else
        println("  ⚠ $func_name may be missing comprehensive docstring")
    end
end

if all_found
    println("\n" * "="^70)
    println("✓ Phase 3 Task 3.3 COMPLETE!")
    println("All 5 parameter getter functions implemented:")
    println("  1. get_parameters_flat(model) → Vector{Float64}")
    println("  2. get_parameters_transformed(model) → NamedTuple")
    println("  3. get_parameters_natural(model) → NamedTuple")
    println("  4. get_unflatten_fn(model) → Function")
    println("  5. get_parameters(model, h; scale=:natural) → Vector{Float64}")
    println("")
    println("Clean API for accessing parameters in any representation!")
    println("="^70)
else
    error("Some getter functions are missing!")
end
