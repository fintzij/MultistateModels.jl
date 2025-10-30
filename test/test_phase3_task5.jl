# Phase 3 Task 3.5 Test: Verify model construction uses parameters_ph
# =======================================================================
#
# Task 3.5: Update multistatemodel() constructors to use parameters_ph
#           from build_hazards() and pass to all model structs
#
# Expected behavior:
# 1. multistatemodel() receives parameters_ph from build_hazards()
# 2. All model types get parameters_ph passed to constructor
# 3. Model structs store parameters_ph correctly
# 4. parameters_ph is properly initialized and accessible

using MultistateModels
using DataFrames

println("\n" * "="^70)
println("PHASE 3 TASK 3.5: Test model construction with parameters_ph")
println("="^70)

# Create test data for different model types
dat_exact = DataFrame(
    id = repeat(1:10, inner=2),
    tstart = repeat([0.0, 1.0], 10),
    tstop = repeat([1.0, 2.0], 10),
    statefrom = repeat([1, 2], 10),
    stateto = repeat([2, 3], 10),
    obstype = ones(Int, 20)  # All exactly observed
)

dat_panel = DataFrame(
    id = repeat(1:10, inner=2),
    tstart = repeat([0.0, 1.0], 10),
    tstop = repeat([1.0, 2.0], 10),
    statefrom = repeat([1, 2], 10),
    stateto = repeat([2, 3], 10),
    obstype = repeat([2, 2], 10)  # All panel data
)

dat_censored = DataFrame(
    id = repeat(1:10, inner=2),
    tstart = repeat([0.0, 1.0], 10),
    tstop = repeat([1.0, 2.0], 10),
    statefrom = repeat([1, 2], 10),
    stateto = repeat([2, 3], 10),
    obstype = repeat([3, 3], 10)  # All censored
)

# Create simple hazard functions
haz1 = ExponentialHazard(:exp1, 1, 2)
haz2 = ExponentialHazard(:exp2, 2, 3)
haz_semi = WeibullHazard(:weib1, 1, 2)  # Semi-Markov

println("\n1. Testing MultistateModel construction...")
println("-" * "="^69)

model_exact = multistatemodel(haz1, haz2; data = dat_exact)

@assert isa(model_exact, MultistateModel) "Should create MultistateModel for exact data"
@assert hasfield(typeof(model_exact), :parameters_ph) "MultistateModel should have parameters_ph field"
@assert !isnothing(model_exact.parameters_ph) "parameters_ph should not be nothing"
@assert haskey(model_exact.parameters_ph, :flat) "parameters_ph should have :flat"
@assert haskey(model_exact.parameters_ph, :transformed) "parameters_ph should have :transformed"
@assert haskey(model_exact.parameters_ph, :natural) "parameters_ph should have :natural"
@assert haskey(model_exact.parameters_ph, :unflatten) "parameters_ph should have :unflatten"

println("✓ MultistateModel constructed with parameters_ph")
println("  - Type: $(typeof(model_exact))")
println("  - parameters_ph.flat length: $(length(model_exact.parameters_ph.flat))")
println("  - parameters_ph.transformed keys: $(keys(model_exact.parameters_ph.transformed))")

println("\n2. Testing MultistateMarkovModel construction...")
println("-" * "="^69)

model_markov = multistatemodel(haz1, haz2; data = dat_panel)

@assert isa(model_markov, MultistateMarkovModel) "Should create MultistateMarkovModel for panel data"
@assert hasfield(typeof(model_markov), :parameters_ph) "MultistateMarkovModel should have parameters_ph field"
@assert !isnothing(model_markov.parameters_ph) "parameters_ph should not be nothing"

println("✓ MultistateMarkovModel constructed with parameters_ph")
println("  - Type: $(typeof(model_markov))")
println("  - parameters_ph fields: $(keys(model_markov.parameters_ph))")

println("\n3. Testing MultistateSemiMarkovModel construction...")
println("-" * "="^69)

model_semi = multistatemodel(haz_semi, haz2; data = dat_panel)

@assert isa(model_semi, MultistateSemiMarkovModel) "Should create MultistateSemiMarkovModel"
@assert hasfield(typeof(model_semi), :parameters_ph) "MultistateSemiMarkovModel should have parameters_ph field"
@assert !isnothing(model_semi.parameters_ph) "parameters_ph should not be nothing"

println("✓ MultistateSemiMarkovModel constructed with parameters_ph")
println("  - Type: $(typeof(model_semi))")
println("  - Hazard types: $([typeof(h) for h in model_semi.hazards])")

println("\n4. Testing MultistateMarkovModelCensored construction...")
println("-" * "="^69)

# Need censoring patterns for censored data
CensoringPatterns = [1 0 1; 0 1 1]  # Example patterns
model_cens_markov = multistatemodel(haz1, haz2; data = dat_censored, CensoringPatterns = CensoringPatterns)

@assert isa(model_cens_markov, MultistateMarkovModelCensored) "Should create MultistateMarkovModelCensored"
@assert hasfield(typeof(model_cens_markov), :parameters_ph) "MultistateMarkovModelCensored should have parameters_ph field"
@assert !isnothing(model_cens_markov.parameters_ph) "parameters_ph should not be nothing"

println("✓ MultistateMarkovModelCensored constructed with parameters_ph")
println("  - Type: $(typeof(model_cens_markov))")
println("  - Censoring patterns: $(size(model_cens_markov.CensoringPatterns))")

println("\n5. Testing MultistateSemiMarkovModelCensored construction...")
println("-" * "="^69)

model_cens_semi = multistatemodel(haz_semi, haz2; data = dat_censored, CensoringPatterns = CensoringPatterns)

@assert isa(model_cens_semi, MultistateSemiMarkovModelCensored) "Should create MultistateSemiMarkovModelCensored"
@assert hasfield(typeof(model_cens_semi), :parameters_ph) "MultistateSemiMarkovModelCensored should have parameters_ph field"
@assert !isnothing(model_cens_semi.parameters_ph) "parameters_ph should not be nothing"

println("✓ MultistateSemiMarkovModelCensored constructed with parameters_ph")
println("  - Type: $(typeof(model_cens_semi))")

println("\n6. Testing parameter consistency across models...")
println("-" * "="^69)

# All Markov models should have same parameters_ph structure for same hazards
@assert length(model_exact.parameters_ph.flat) == length(model_markov.parameters_ph.flat) "Same hazards should have same flat length"
@assert keys(model_exact.parameters_ph.transformed) == keys(model_markov.parameters_ph.transformed) "Same hazards should have same keys"

println("✓ Parameter structures consistent across model types")
println("  - Flat parameter count matches: $(length(model_exact.parameters_ph.flat))")
println("  - Transformed keys match: $(keys(model_exact.parameters_ph.transformed))")

println("\n7. Testing parameters_ph accessibility...")
println("-" * "="^69)

# Test that we can access and use parameters_ph
flat_params = model_exact.parameters_ph.flat
natural_params = model_exact.parameters_ph.natural
unflatten_fn = model_exact.parameters_ph.unflatten

@assert length(flat_params) > 0 "Flat parameters should not be empty"
@assert length(natural_params) > 0 "Natural parameters should not be empty"
@assert isa(unflatten_fn, Function) "Unflatten should be a function"

# Test unflatten roundtrip
reconstructed = unflatten_fn(flat_params)
@assert keys(reconstructed) == keys(model_exact.parameters_ph.transformed) "Unflatten should reconstruct same keys"

println("✓ parameters_ph accessible and functional")
println("  - Flat parameters: $(flat_params)")
println("  - Natural parameters: $(natural_params)")
println("  - Unflatten works: $(keys(reconstructed) == keys(model_exact.parameters_ph.transformed))")

println("\n" * "="^70)
println("✓ Phase 3 Task 3.5 COMPLETE!")
println("="^70)
println("\nSummary:")
println("  ✓ MultistateModel constructed with parameters_ph")
println("  ✓ MultistateMarkovModel constructed with parameters_ph")
println("  ✓ MultistateSemiMarkovModel constructed with parameters_ph")
println("  ✓ MultistateMarkovModelCensored constructed with parameters_ph")
println("  ✓ MultistateSemiMarkovModelCensored constructed with parameters_ph")
println("  ✓ Parameter structures consistent across model types")
println("  ✓ parameters_ph accessible and functional")
println("\nAll 5 model types successfully store and provide parameters_ph!")
