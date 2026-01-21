# MCEM Spline Diagnostic: Understand why boundary coefficient is overestimated
using BSplineKit
using LinearAlgebra
using Statistics
using QuadGK

println("=" ^ 60)
println("MCEM SPLINE BOUNDARY COEFFICIENT DIAGNOSTIC")
println("=" ^ 60)
println()

# Setup: same as MultistateModels spline builder
degree = 3
intknots = [5.0, 10.0]
bknots = [0.0, 15.0]
allknots = unique(sort([bknots[1]; intknots; bknots[2]]))
t_max = 15.0

# Build recombined basis
B_orig = BSplineBasis(BSplineOrder(degree + 1), copy(allknots))
B = RecombinedBSplineBasis(B_orig, Derivative(1))
nbasis = length(B)

println("Basis setup:")
println("  Knots: $allknots")
println("  Degree: $degree")
println("  N basis functions: $nbasis")
println()

# True coefficients (from test fixture)
true_coefs = [0.08, 0.10, 0.14, 0.18]

# Create spline function
function make_hazard(coefs)
    spline = Spline(B, coefs)
    spline_ext = SplineExtrapolation(spline, BSplineKit.SplineExtrapolations.Flat())
    return t -> spline_ext(t)
end

h_true = make_hazard(true_coefs)

println("=" ^ 60)
println("ANALYSIS 1: Integral contributions by coefficient")
println("=" ^ 60)
println()

# For each coefficient, compute its contribution to the integral ∫h(t)dt
# This tells us how much each coefficient contributes to the cumulative hazard

function coefficient_contribution(coefs, j, t_range)
    # Zero all coefficients except j
    test_coefs = zeros(nbasis)
    test_coefs[j] = coefs[j]
    h = make_hazard(test_coefs)
    integral, _ = quadgk(h, t_range[1], t_range[2])
    return integral
end

println("Contribution to ∫₀^15 h(t)dt for each coefficient:")
total_integral, _ = quadgk(h_true, 0.0, t_max)
println("  Total integral: $(round(total_integral, digits=4))")
for j in 1:nbasis
    contrib = coefficient_contribution(true_coefs, j, (0.0, t_max))
    pct = 100 * contrib / total_integral
    println("  coef_$j = $(true_coefs[j]): integral = $(round(contrib, digits=4)) ($(round(pct, digits=1))%)")
end
println()

println("=" ^ 60)
println("ANALYSIS 2: Basis function support")
println("=" ^ 60)
println()

# For each basis function, where is it non-zero?
println("Basis function support regions:")
for j in 1:nbasis
    # Find where basis function is > 1e-10
    times_active = Float64[]
    for t in 0.0:0.1:15.0
        if B[j](t) > 1e-6
            push!(times_active, t)
        end
    end
    if !isempty(times_active)
        println("  B_$j: t ∈ [$(minimum(times_active)), $(maximum(times_active))]")
    end
end
println()

println("=" ^ 60)
println("ANALYSIS 3: Score function (∂logL/∂coef) sensitivity")
println("=" ^ 60)
println()

# The log-likelihood for a transition at time t with hazard h(t) is:
# log(h(t)) - ∫₀^t h(s)ds
# 
# The gradient ∂logL/∂coef_j = B_j(t)/h(t) - ∫₀^t B_j(s) ds
#
# At the boundary t=15, B_4(15) = 1 and all others = 0.
# So ∂logL/∂coef_4 = 1/h(15) - ∫₀^15 B_4(s) ds
# But ∂logL/∂coef_j = 0 - ∫₀^15 B_j(s) ds for j ≠ 4

# This creates an asymmetry: a transition at t=15 ONLY provides information about coef_4!

function grad_loglik_transition(coefs, t_trans)
    h = make_hazard(coefs)
    h_t = h(t_trans)
    
    grad = zeros(nbasis)
    for j in 1:nbasis
        # ∂h(t)/∂coef_j = B_j(t)
        dh_t = B[j](t_trans)
        
        # ∂H(0,t)/∂coef_j = ∫₀^t B_j(s) ds
        dH_t, _ = quadgk(s -> B[j](s), 0.0, t_trans)
        
        # ∂logL/∂coef_j = dh_t / h_t - dH_t
        grad[j] = dh_t / h_t - dH_t
    end
    return grad
end

println("Score at different transition times (using true coefficients):")
for t in [2.5, 5.0, 7.5, 10.0, 12.5, 14.0, 15.0]
    grad = grad_loglik_transition(true_coefs, t)
    println("  t = $t:")
    for j in 1:nbasis
        println("    ∂logL/∂coef_$j = $(round(grad[j], digits=4))")
    end
end
println()

println("=" ^ 60)
println("ANALYSIS 4: Fisher Information contribution by region")
println("=" ^ 60)
println()

# The diagonal of Fisher info is approximately:
# I_jj ≈ E[(∂logL/∂coef_j)²]
# 
# Transitions near the boundary only contribute to coef_4's information.
# If MCEM samples transitions uniformly, the boundary region [12.5, 15] contributes
# only to coef_4, creating identifiability issues.

println("Basis values in last panel interval [12.5, 15.0]:")
for t in 12.5:0.5:15.0
    vals = [B[j](t) for j in 1:nbasis]
    println("  t = $t: $(round.(vals, digits=4))")
end
println()

# The problem: at t=15, B_4=1, all others=0.
# A transition at exactly t=15 gives ALL its hazard information to coef_4.

println("=" ^ 60)
println("ANALYSIS 5: What overestimated coef_4 means for hazard")
println("=" ^ 60)
println()

# With overestimated coef_4, the hazard at high times is wrong
bad_coefs = [0.0848, 0.1866, 0.0542, 0.867]  # From test results
h_bad = make_hazard(bad_coefs)

println("Hazard comparison (true vs estimated with 382% coef_4 error):")
for t in [0.0, 5.0, 10.0, 12.5, 14.0, 15.0]
    h_t = h_true(t)
    h_b = h_bad(t)
    ratio = h_b / h_t
    println("  t=$t: h_true=$(round(h_t, digits=4)), h_est=$(round(h_b, digits=4)), ratio=$(round(ratio, digits=2))")
end
println()

# Integral comparison
int_true, _ = quadgk(h_true, 0.0, 15.0)
int_bad, _ = quadgk(h_bad, 0.0, 15.0)
println("Cumulative hazard [0, 15]:")
println("  True: $(round(int_true, digits=4))")
println("  Estimated: $(round(int_bad, digits=4))")
println("  Ratio: $(round(int_bad/int_true, digits=2))")
println()

println("=" ^ 60)
println("KEY INSIGHT")
println("=" ^ 60)
println("""
The RecombinedBSplineBasis with Derivative(1) boundary condition creates:
  - At t=0: only B_1 is non-zero (B_1 = 1)
  - At t=15: only B_4 is non-zero (B_4 = 1)

This means:
  1. The hazard at the boundary equals the boundary coefficient exactly
  2. Transitions AT the boundary provide information ONLY about that coefficient
  3. MCEM paths may have transitions ending at exactly t=15 (panel endpoint)
  4. These boundary transitions bias coef_4 estimation

POTENTIAL FIXES:
  1. Use a regular B-spline basis without recombination (no Derivative(1))
  2. Exclude transitions exactly at boundary from likelihood
  3. Extend the boundary knots beyond the observation window
  4. Use different extrapolation (e.g., "linear" or "none")
""")
