# Performance profiling for MultistateModels
# Run this script to identify bottlenecks in Markov and phase-type model fitting

using MultistateModels
using DataFrames
using Distributions
using Random
using BenchmarkTools
using Profile
using LinearAlgebra  # For Diagonal

Random.seed!(12345)

# =============================================================================
# SECTION 1: Create test models for profiling
# =============================================================================

println("=" ^ 60)
println("MultistateModels Performance Profiling")
println("=" ^ 60)

# --- 1A: Simple 3-state Markov model (illness-death) ---
println("\n--- Setting up 3-state illness-death model ---")

# Transition matrix: 1 → 2 (illness), 1 → 3 (death), 2 → 3 (death)
tmat = [0 1 2;
        0 0 3;
        0 0 0]

n_subjects = 500
n_obs_per_subject = 8

# Generate panel data
ids = repeat(1:n_subjects, inner=n_obs_per_subject)
times = repeat(collect(0.0:1.0:(n_obs_per_subject-1)), outer=n_subjects)

# Simple covariate
x1 = repeat(randn(n_subjects), inner=n_obs_per_subject)

# Simulate states (simplified - just random valid transitions)
states = Int[]
for i in 1:n_subjects
    state = 1
    for j in 1:n_obs_per_subject
        push!(states, state)
        if state == 1
            r = rand()
            if r < 0.1
                state = 2
            elseif r < 0.15
                state = 3
            end
        elseif state == 2
            if rand() < 0.15
                state = 3
            end
        end
        # state 3 is absorbing
    end
end

markov_data = DataFrame(
    id = ids,
    tstart = times,
    tstop = times .+ 1.0,
    statefrom = states,
    stateto = circshift(states, -1),  # Will fix below
    obstype = fill(2, length(ids)),  # Panel data
    x1 = x1
)

# Fix stateto for each subject's last observation
for i in 1:n_subjects
    last_idx = i * n_obs_per_subject
    markov_data.stateto[last_idx] = markov_data.statefrom[last_idx]
end

# Fix statefrom/stateto alignment within subjects
for i in 1:n_subjects
    start_idx = (i-1) * n_obs_per_subject + 1
    end_idx = i * n_obs_per_subject
    for j in (start_idx+1):end_idx
        markov_data.statefrom[j] = markov_data.stateto[j-1]
    end
end

# Create hazards - exponential with covariate
h12 = Hazard(@formula(0 ~ 1 + x1), "exp", 1, 2)
h13 = Hazard(@formula(0 ~ 1 + x1), "exp", 1, 3)
h23 = Hazard(@formula(0 ~ 1 + x1), "exp", 2, 3)

println("Creating Markov model with $n_subjects subjects, $(nrow(markov_data)) observations...")
markov_model = multistatemodel(h12, h13, h23; data=markov_data)

# =============================================================================
# SECTION 2: Benchmark Markov model likelihood
# =============================================================================

println("\n--- Benchmarking Markov model likelihood ---")

# Get parameters
params_markov = MultistateModels.get_parameters_flat(markov_model)

# Create MPanelData (needs books for TPM mapping)
books_markov = MultistateModels.build_tpm_mapping(markov_model.data)
markov_panel_data = MultistateModels.MPanelData(markov_model, books_markov)

# Benchmark single likelihood evaluation
println("\nSingle likelihood evaluation:")
@btime MultistateModels.loglik_markov($params_markov, $markov_panel_data; neg=true)

# Profile detailed breakdown
println("\nProfiling likelihood computation...")
Profile.clear()
@profile for _ in 1:100
    MultistateModels.loglik_markov(params_markov, markov_panel_data; neg=true)
end

# Print profile results
println("\nTop functions by time (Markov likelihood):")
Profile.print(format=:flat, mincount=50, sortedby=:count)

# =============================================================================
# SECTION 3: Benchmark matrix exponential specifically
# =============================================================================

println("\n" * "=" ^ 60)
println("Matrix Exponential Performance")
println("=" ^ 60)

using ExponentialUtilities

# Test different matrix sizes
for S in [3, 5, 8, 10, 15]
    println("\n--- Matrix size: $S × $S ---")
    
    # Create rate matrix
    Q = randn(S, S)
    Q = Q - Diagonal(sum(Q, dims=2)[:])
    
    # Different time scales
    t = 1.0
    
    # Allocate cache
    cache = ExponentialUtilities.alloc_mem(similar(Q), MultistateModels.ExpMethodGeneric())
    P = similar(Q)
    
    # Benchmark
    println("Matrix exponential (ExpMethodGeneric):")
    @btime copyto!($P, exponential!($Q * $t, $(MultistateModels.ExpMethodGeneric()), $cache))
    
    # Compare with different methods if available
    println("Matrix exponential (exp - Julia built-in):")
    @btime exp($Q * $t)
end

# =============================================================================
# SECTION 4: Benchmark Markov model fitting
# =============================================================================

println("\n" * "=" ^ 60)
println("Markov Model Fitting Performance")
println("=" ^ 60)

println("\nFitting Markov model (unconstrained, L-BFGS)...")
fit_time = @elapsed fitted_markov = fit(markov_model; verbose=false)
println("Fit time: $(round(fit_time, digits=2)) seconds")

# Profile the fit
println("\nProfiling fit() function...")
Profile.clear()
@profile fit(markov_model; verbose=false)
println("\nTop functions by time (Markov fit):")
Profile.print(format=:flat, mincount=30, sortedby=:count)

# =============================================================================
# SECTION 5: Phase-type model performance (if available)
# =============================================================================

println("\n" * "=" ^ 60)
println("Phase-Type Model Performance")
println("=" ^ 60)

# Simple 2-state model for phase-type expansion
tmat_pt = [0 1;
           0 0]

# Generate simple survival data
n_pt = 200
pt_data = DataFrame(
    id = 1:n_pt,
    tstart = zeros(n_pt),
    tstop = rand(Exponential(5.0), n_pt),
    statefrom = ones(Int, n_pt),
    stateto = fill(2, n_pt),
    obstype = fill(2, n_pt)  # Panel data
)

# Configure phase-type model with 3 phases
config = MultistateModels.PhaseTypeConfig(
    n_phases = Dict(1 => 3),  # 3 phases for state 1
    cv_init = Dict(1 => 0.8),
    mean_init = Dict(1 => 5.0)
)

println("\nBuilding phase-type model with 3 phases...")
build_time = @elapsed pt_model = MultistateModels.build_phasetype_model(tmat_pt, config; data=pt_data)
println("Build time: $(round(build_time, digits=4)) seconds")

# Benchmark phase-type likelihood
books_pt = MultistateModels.build_tpm_mapping(pt_model.data)
pt_panel_data = MultistateModels.MPanelData(pt_model, books_pt)
params_pt = MultistateModels.get_parameters_flat(pt_model)

println("\nBenchmarking phase-type likelihood:")
@btime MultistateModels.loglik_markov($params_pt, $pt_panel_data; neg=true)

# Profile phase-type likelihood
println("\nProfiling phase-type likelihood...")
Profile.clear()
@profile for _ in 1:100
    MultistateModels.loglik_markov(params_pt, pt_panel_data; neg=true)
end
println("\nTop functions by time (phase-type likelihood):")
Profile.print(format=:flat, mincount=30, sortedby=:count)

# Fit phase-type model
println("\nFitting phase-type model...")
pt_fit_time = @elapsed fitted_pt = fit(pt_model; verbose=false)
println("Fit time: $(round(pt_fit_time, digits=2)) seconds")

# =============================================================================
# SECTION 6: Memory allocation analysis
# =============================================================================

println("\n" * "=" ^ 60)
println("Memory Allocation Analysis")
println("=" ^ 60)

println("\nMarkov likelihood memory allocations:")
@btime MultistateModels.loglik_markov($params_markov, $markov_panel_data)

println("\nPhase-type likelihood memory allocations:")
@btime MultistateModels.loglik_markov($params_pt, $pt_panel_data)

# =============================================================================
# SECTION 7: Summary and recommendations
# =============================================================================

println("\n" * "=" ^ 60)
println("Performance Summary")
println("=" ^ 60)

println("""
Key areas to investigate for optimization:

1. **Matrix Exponential**: 
   - Currently using ExpMethodGeneric
   - For small matrices (< 10×10), consider scaling and squaring
   - For larger matrices, consider Padé approximation

2. **Memory Allocations**:
   - Reduce DataFrame row access allocations
   - Pre-allocate working arrays in likelihood loops
   - Use views instead of copies where possible

3. **Forward Algorithm** (for censored data):
   - The nested loops in the forward algorithm can be expensive
   - Consider BLAS operations for matrix-vector products

4. **Covariate Extraction**:
   - DataFrame row access is allocation-heavy
   - Consider caching covariate values per subject

5. **AD (ForwardDiff) overhead**:
   - Dual number arithmetic adds overhead
   - Consider chunking strategies for large parameter vectors
""")
