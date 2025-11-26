# Example: Semi-Markov Model with Mixed Hazard Types
# This script demonstrates creating a semi-markov model with different hazard families
# (exponential, Weibull, Gompertz) to show the flexibility of the package

using DataFrames
using MultistateModels
using Distributions
using Random

Random.seed!(123)

# ============================================================================
# STEP 1: Create simulated illness-death data
# ============================================================================
# States: 1 = Healthy, 2 = Ill, 3 = Dead
# Transitions: 1→2 (illness), 1→3 (direct death), 2→3 (death after illness)

n_subjects = 100
max_time = 10.0

# Generate data
data_list = []

for i in 1:n_subjects
    age = rand(40:70)
    trt = rand([0, 1])
    
    # Subject starts healthy at time 0
    t_current = 0.0
    state_current = 1
    
    # Simulate illness time (1→2) - Weibull with age effect
    λ_12 = exp(-1.5 + 0.02 * age - 0.5 * trt)
    α_12 = 1.5  # Shape parameter
    t_illness = rand(Weibull(α_12, 1/λ_12))
    
    # Simulate direct death time (1→3) - Exponential with age effect
    λ_13 = exp(-3.0 + 0.03 * age)
    t_direct_death = rand(Exponential(1/λ_13))
    
    # Simulate death after illness (2→3) - Gompertz
    λ_23 = exp(-2.0 + 0.5 * trt)
    γ_23 = 0.1  # Gompertz shape
    # Approximate Gompertz sampling (simplified)
    t_death_after_illness = rand(Exponential(1/λ_23))
    
    # Determine what happens
    if t_illness < min(t_direct_death, max_time)
        # Gets ill first
        push!(data_list, (id=i, tstart=0.0, tstop=t_illness, 
                         statefrom=1, stateto=2, obstype=1, age=age, trt=trt))
        
        if t_illness + t_death_after_illness < max_time
            # Dies after illness
            push!(data_list, (id=i, tstart=t_illness, tstop=t_illness + t_death_after_illness,
                             statefrom=2, stateto=3, obstype=1, age=age, trt=trt))
        else
            # Censored while ill
            push!(data_list, (id=i, tstart=t_illness, tstop=max_time,
                             statefrom=2, stateto=2, obstype=2, age=age, trt=trt))
        end
    elseif t_direct_death < max_time
        # Direct death before illness
        push!(data_list, (id=i, tstart=0.0, tstop=t_direct_death,
                         statefrom=1, stateto=3, obstype=1, age=age, trt=trt))
    else
        # Censored while healthy
        push!(data_list, (id=i, tstart=0.0, tstop=max_time,
                         statefrom=1, stateto=1, obstype=2, age=age, trt=trt))
    end
end

dat = DataFrame(data_list)

println("=" ^ 70)
println("SIMULATED DATA SUMMARY")
println("=" ^ 70)
println("Number of subjects: $n_subjects")
println("Maximum follow-up time: $max_time")
println("\nData structure:")
println(first(dat, 10))
println("\nObservation type counts:")
println(combine(groupby(dat, :obstype), nrow => :count))
println("\nTransition counts:")
println(combine(groupby(dat, [:statefrom, :stateto]), nrow => :count))

# ============================================================================
# STEP 2: Define hazards with different families
# ============================================================================

println("\n" * "=" ^ 70)
println("DEFINING MIXED HAZARD MODEL")
println("=" ^ 70)

# 1→2: Illness onset - WEIBULL hazard with age and treatment effects
# Weibull is semi-Markov (time-dependent baseline hazard)
h12 = Hazard(@formula(0 ~ 1 + age + trt), "wei", 1, 2)
println("✓ Transition 1→2 (Healthy → Ill): Weibull (semi-Markov)")
println("  Covariates: age, treatment")

# 1→3: Direct death from healthy - EXPONENTIAL hazard with age effect
# Exponential is Markov (time-independent baseline hazard)
h13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
println("✓ Transition 1→3 (Healthy → Dead): Exponential (Markov)")
println("  Covariates: age")

# 2→3: Death after illness - GOMPERTZ hazard with treatment effect
# Gompertz is semi-Markov (accelerating/decelerating hazard)
h23 = Hazard(@formula(0 ~ 1 + trt), "gom", 2, 3)
println("✓ Transition 2→3 (Ill → Dead): Gompertz (semi-Markov)")
println("  Covariates: treatment")

# ============================================================================
# STEP 3: Build the model
# ============================================================================

println("\n" * "=" ^ 70)
println("BUILDING MULTISTATE MODEL")
println("=" ^ 70)

model = multistatemodel(h12, h13, h23; data = dat)

println("✓ Model built successfully!")
println("\nModel type: $(typeof(model))")
println("This is a SEMI-MARKOV model because it contains at least one")
println("time-dependent hazard (Weibull or Gompertz).")

# ============================================================================
# STEP 4: Inspect the model structure
# ============================================================================

println("\n" * "=" ^ 70)
println("MODEL STRUCTURE")
println("=" ^ 70)

println("\nTransition matrix:")
println(model.tmat)
println("  (rows = from state, columns = to state)")
println("  (non-zero entries are transition numbers)")

println("\nHazard information:")
for (i, h) in enumerate(model.hazards)
    println("\nTransition $i ($(h.statefrom) → $(h.stateto)):")
    println("  Family: $(h.family)")
    println("  Type: $(typeof(h))")
    println("  Parameters: $(h.parnames)")
    println("  Has covariates: $(h.has_covariates)")
end

println("\nTotal hazards (by origin state):")
for (i, th) in enumerate(model.totalhazards)
    println("  State $i: $(typeof(th))")
    if isa(th, MultistateModels._TotalHazardTransient)
        println("    Possible transitions: $(th.hazard_indices)")
    end
end

# ============================================================================
# STEP 5: Check parameters
# ============================================================================

println("\n" * "=" ^ 70)
println("PARAMETER STRUCTURE")
println("=" ^ 70)

println("\nParameters for each hazard:")
for (i, pars) in enumerate(model.parameters)
    h = model.hazards[i]
    println("\nHazard $i ($(h.statefrom) → $(h.stateto), $(h.family)):")
    println("  Parameter names: $(h.parnames)")
    println("  Initial values: $(pars)")
    println("  Number of parameters: $(length(pars))")
end

# ============================================================================
# STEP 6: Demonstrate the semi-Markov property
# ============================================================================

println("\n" * "=" ^ 70)
println("SEMI-MARKOV PROPERTY DEMONSTRATION")
println("=" ^ 70)

println("\nIn a semi-Markov model, hazards can depend on time since entry to state.")
println("\nExample for Weibull hazard (1→2):")
println("  At time t=0 (just entered state 1):")
println("    The hazard depends on baseline Weibull and covariates")
println("  At time t=5 (been in state 1 for 5 units):")
println("    The hazard has changed according to Weibull shape parameter")

# Get a sample row for demonstration
sample_row = dat[1, :]
println("\nSample subject (ID=$(sample_row.id), age=$(sample_row.age), trt=$(sample_row.trt)):")

# Call hazard at different times to show time-dependence
times = [0.0, 1.0, 2.0, 5.0, 10.0]
println("\nWeibull hazard 1→2 at different sojourn times:")
for t in times
    # For Weibull, hazard changes with time
    haz_val = MultistateModels.call_haz(t, model.parameters[1], sample_row, model.hazards[1])
    println("  t=$t: h(t) = $(round(exp(haz_val), digits=6))")
end

println("\nExponential hazard 1→3 at different sojourn times:")
for t in times
    # For Exponential, hazard is constant (Markov property)
    haz_val = MultistateModels.call_haz(t, model.parameters[2], sample_row, model.hazards[2])
    println("  t=$t: h(t) = $(round(exp(haz_val), digits=6))  (constant!)")
end

# ============================================================================
# STEP 7: Summary
# ============================================================================

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nThis example demonstrates:")
println("  1. Creating a semi-Markov illness-death model")
println("  2. Mixing different hazard families in one model:")
println("     - Exponential (Markov): constant baseline hazard")
println("     - Weibull (semi-Markov): time-dependent baseline hazard")
println("     - Gompertz (semi-Markov): accelerating/decelerating hazard")
println("  3. Including covariate effects on transitions")
println("  4. The model is classified as semi-Markov if ANY hazard")
println("     is time-dependent (Weibull, Gompertz, or splines)")

println("\n" * "=" ^ 70)
println("MODEL READY FOR ESTIMATION")
println("=" ^ 70)
println("\nNext steps:")
println("  - Use fit_msm() to estimate parameters")
println("  - Use sample_paths() for path imputation (if panel/censored data)")
println("  - Use simulate() to generate predictions")

println("\n✓ Script completed successfully!")
