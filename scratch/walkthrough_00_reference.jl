# Internal Functions Reference
# This script documents internal (non-exported) functions that can be accessed
# during the walkthrough using the MultistateModels. prefix

using MultistateModels
using DataFrames

println("=" ^ 70)
println("INTERNAL FUNCTIONS REFERENCE")
println("=" ^ 70)

println("\n📚 KEY INTERNAL FUNCTIONS FOR WALKTHROUGH")
println("=" ^ 70)

# ============================================================================
# TYPE CONSTRUCTORS (Internal, not exported)
# ============================================================================
println("\n1. TYPE CONSTRUCTORS (Not Exported)")
println("-" ^ 70)

println("""
MultistateModels.Hazard(from, to; family="exp", formula=nothing)
  - Creates hazard specification (exported as Hazard in module)
  - Returns: MarkovHazard, SemiMarkovHazard, or SplineHazard
  - Arguments:
    * from::Int - origin state
    * to::Int - destination state
    * family::String - "exp", "wei", "gom", "sp"
    * formula::FormulaTerm - @formula(0 ~ covariate1 + covariate2)
  - Example:
    haz = MultistateModels.Hazard(1, 2, family="exp", formula=@formula(0 ~ age))
""")

# ============================================================================
# PARAMETER INITIALIZATION (Internal)
# ============================================================================
println("\n2. PARAMETER INITIALIZATION")
println("-" ^ 70)

println("""
MultistateModels.init_par(hazard::Union{MarkovHazard,SemiMarkovHazard,SplineHazard}, crude_log_rate=0)
  - Initialize parameters for a hazard
  - Returns: Vector of initial parameter values
  - Dispatches on hazard.family:
    * "exp" → [crude_log_rate] or [crude_log_rate, zeros(ncovar)...]
    * "wei" → [0.0, crude_log_rate] or [0.0, crude_log_rate, zeros(ncovar)...]
    * "gom" → [0.0, crude_log_rate] or [0.0, crude_log_rate, zeros(ncovar)...]
  - Example:
    init_params = MultistateModels.init_par(haz, 0.0)
""")

# ============================================================================
# COVARIATE EXTRACTION (Internal)
# ============================================================================
println("\n3. COVARIATE EXTRACTION (Critical for Name-Based Matching)")
println("-" ^ 70)

println("""
MultistateModels.extract_covar_names(parnames::Vector{Symbol})
  - Extract covariate parameter names from full parameter list
  - Filters out baseline parameters (intercept, shape, scale)
  - Returns: Vector{Symbol} of covariate names
  - Example:
    covar_names = MultistateModels.extract_covar_names([:intercept, :age, :sex])
    # Returns: [:age, :sex]

MultistateModels.extract_covariates(subjdat::Union{DataFrameRow,DataFrame}, parnames::Vector{Symbol})
  - Extract covariates from data row as NamedTuple
  - Only extracts columns matching covariate names
  - Returns: NamedTuple with covariate values
  - Example:
    row = data[1, :]  # Has columns: id, tstart, tstop, age, sex, treatment
    covars = MultistateModels.extract_covariates(row, [:age, :sex])
    # Returns: (age = 45.2, sex = 1.0)
    # Access: covars.age, covars.sex
""")

# ============================================================================
# HAZARD TYPE HIERARCHY (Internal)
# ============================================================================
println("\n4. HAZARD TYPE HIERARCHY")
println("-" ^ 70)

println("""
Abstract Types:
  - MultistateModels.AbstractHazard (top level)
  - MultistateModels.AbstractMarkovHazard <: AbstractHazard
  - MultistateModels.AbstractSemiMarkovHazard <: AbstractHazard
  - MultistateModels.AbstractSplineHazard <: AbstractHazard

Concrete Types (all have same structure):
  - MultistateModels.MarkovHazard <: AbstractMarkovHazard
    * Used for: Exponential (family="exp")
    * Memoryless: hazard depends only on current state
  
  - MultistateModels.SemiMarkovHazard <: AbstractSemiMarkovHazard
    * Used for: Weibull (family="wei"), Gompertz (family="gom")
    * Clock-reset: hazard depends on time since state entry
  
  - MultistateModels.SplineHazard <: AbstractSplineHazard
    * Used for: Splines (family="sp")
    * Flexible baseline hazard

Common Fields (all types):
  - hazname::Symbol - unique identifier
  - from::Int - origin state
  - to::Int - destination state
  - family::String - hazard family ("exp", "wei", "gom", "sp")
  - parnames::Vector{Symbol} - parameter names
  - npar_baseline::Int - number of baseline parameters
  - npar_total::Int - total parameters (baseline + covariates)
  - hazard_fn::Function - runtime-generated hazard function
  - cumhaz_fn::Function - runtime-generated cumulative hazard function
  - has_covariates::Bool - whether covariates present
""")

# ============================================================================
# RUNTIME-GENERATED FUNCTIONS (Internal)
# ============================================================================
println("\n5. RUNTIME-GENERATED HAZARD FUNCTIONS")
println("-" ^ 70)

println("""
Each hazard has two runtime-generated functions:

hazard.hazard_fn(t, params, covars; give_log=true)
  - Evaluate hazard at time t
  - Arguments:
    * t::Real - time point
    * params::NamedTuple - named parameters
    * covars::NamedTuple - named covariates
    * give_log::Bool - return log-hazard if true
  - Example:
    h = haz.hazard_fn(1.0, (intercept=-1.0, age=0.02), (age=45.0,); give_log=false)

hazard.cumhaz_fn(lb, ub, params, covars)
  - Evaluate cumulative hazard over interval
  - Arguments:
    * lb::Real - interval lower bound
    * ub::Real - interval upper bound
    * params::NamedTuple - named parameters
    * covars::NamedTuple - named covariates
  - Returns: ∫_{lb}^{ub} h(t) dt
  - Example:
    cumh = haz.cumhaz_fn(0.0, 1.0, (intercept=-1.0,), NamedTuple())
""")

# ============================================================================
# MODEL STRUCTURE (Exported)
# ============================================================================
println("\n6. MODEL STRUCTURE")
println("-" ^ 70)

println("""
multistatemodel(hazards...; data::DataFrame)
  - Create multi-state model from hazards
  - Arguments:
    * hazards... - variable number of Hazard objects
    * data::DataFrame - must have columns: id, tstart, tstop, statefrom, stateto
  - Returns: MultistateModel object
  - Example:
    model = multistatemodel(haz1, haz2, haz3; data=dat)

MultistateModel fields (partial list):
  - hazards::Vector{<:AbstractHazard} - all hazards
  - data::DataFrame - subject data
  - nsubj::Int - number of subjects
  - npar::Int - total parameters across all hazards
  - paths - transition information
  - obstimes - observation times
""")

# ============================================================================
# HELPER FUNCTIONS FOR EXPLORATION
# ============================================================================
println("\n7. HELPER FUNCTIONS FOR EXPLORATION")
println("-" ^ 70)

println("""
To explore a hazard object:
  - typeof(haz) - check concrete type
  - fieldnames(typeof(haz)) - list all fields
  - haz.family - hazard family
  - haz.parnames - parameter names
  - haz.has_covariates - covariate presence
  - methods(haz.hazard_fn) - check generated function signature

To explore a model:
  - typeof(model) - MultistateModel
  - fieldnames(typeof(model)) - list all fields
  - length(model.hazards) - number of hazards
  - model.npar - total parameters
  - model.nsubj - number of subjects
  
To check type hierarchy:
  - typeof(haz) <: MultistateModels.MarkovHazard
  - typeof(haz) <: MultistateModels.AbstractHazard
""")

# ============================================================================
# COMMON PATTERNS
# ============================================================================
println("\n8. COMMON PATTERNS FOR WALKTHROUGH")
println("-" ^ 70)

println("""
Pattern 1: Create and test simple hazard
  haz = MultistateModels.Hazard(1, 2, family="exp")
  params = (intercept = -1.0,)
  h = haz.hazard_fn(1.0, params, NamedTuple(); give_log=false)

Pattern 2: Create hazard with covariates
  haz = MultistateModels.Hazard(1, 2, family="exp", formula=@formula(0 ~ age + sex))
  covar_names = MultistateModels.extract_covar_names(haz.parnames)
  row = data[1, :]
  covars = MultistateModels.extract_covariates(row, covar_names)
  params = (intercept = -1.0, age = 0.02, sex = 0.5)
  h = haz.hazard_fn(1.0, params, covars; give_log=false)

Pattern 3: Initialize parameters
  init_params = MultistateModels.init_par(haz, 0.0)

Pattern 4: Build multi-state model
  haz1 = MultistateModels.Hazard(1, 2, family="exp")
  haz2 = MultistateModels.Hazard(2, 3, family="wei")
  model = multistatemodel(haz1, haz2; data=dat)

Pattern 5: Different covariates per hazard
  haz_1_2 = MultistateModels.Hazard(1, 2, family="exp", formula=@formula(0 ~ age + sex))
  haz_2_3 = MultistateModels.Hazard(2, 3, family="exp", formula=@formula(0 ~ treatment))
  # Each hazard will extract only its covariates by name!
""")

# ============================================================================
# DEMONSTRATION
# ============================================================================
println("\n9. QUICK DEMONSTRATION")
println("-" ^ 70)

# Create simple hazard
haz = MultistateModels.Hazard(1, 2, family="exp")
println("Created hazard: ", typeof(haz))
println("  Family: ", haz.family)
println("  Parameters: ", haz.parnames)
println("  Has covariates: ", haz.has_covariates)

# Test evaluation
params = (intercept = -1.0,)
h = haz.hazard_fn(1.0, params, NamedTuple(); give_log=false)
println("\nEvaluated h(1.0) with intercept=-1.0: ", h)
println("  Expected: ", exp(-1.0))
println("  Match: ", isapprox(h, exp(-1.0)))

# Initialize
init = MultistateModels.init_par(haz, 0.0)
println("\nInitialized parameters: ", init)

println("\n" * "=" ^ 70)
println("REFERENCE COMPLETE")
println("=" ^ 70)
println("\nNow you can walk through the package code line by line!")
println("All internal functions are accessible via MultistateModels. prefix")
println("\nStart with: include(\"walkthrough_01_basic_model.jl\")")
