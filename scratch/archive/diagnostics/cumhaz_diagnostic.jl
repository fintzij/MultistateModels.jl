# Cumulative hazard diagnostic for spline with RecombinedBSplineBasis
using BSplineKit

println("=" ^ 60)
println("CUMULATIVE HAZARD DIAGNOSTIC")
println("=" ^ 60)
println()

# Setup
degree = 3
allknots = [0.0, 5.0, 10.0, 15.0]
B_orig = BSplineBasis(BSplineOrder(degree + 1), copy(allknots))
B = RecombinedBSplineBasis(B_orig, Derivative(1))

# True coefficients
true_coefs = [0.08, 0.10, 0.14, 0.18]
spline = Spline(B, true_coefs)
spline_ext = SplineExtrapolation(spline, BSplineKit.SplineExtrapolations.Flat())

# Cumulative hazard via integral
cumhaz_spline = integral(spline_ext.spline)

println("Cumulative hazard H(0, t) at panel times:")
for t in [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    H = cumhaz_spline(t) - cumhaz_spline(0.0)
    println("  H(0, $t) = $(round(H, digits=4))")
end
println()

println("Incremental cumulative hazard H(t_prev, t) between panels:")
times = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
for i in 2:length(times)
    t0, t1 = times[i-1], times[i]
    dH = cumhaz_spline(t1) - cumhaz_spline(t0)
    println("  H($t0, $t1) = $(round(dH, digits=4))")
end
println()

# Now check what BSplineKit gives us for the integral at the boundary
println("Raw cumhaz_spline values:")
for t in [0.0, 5.0, 10.0, 14.0, 14.5, 14.9, 14.99, 15.0]
    println("  cumhaz_spline($t) = $(cumhaz_spline(t))")
end
println()

# Check derivative of cumhaz_spline at boundary
println("Checking derivative of integrated spline at boundary:")
println("  d(cumhaz)/dt|_{t=15-} = spline(15) = $(spline_ext(15.0))")

# This is fine - integral of constant hazard grows linearly
# But what happens if we evaluate BEYOND 15?
println()
println("Behavior beyond boundary (should use flat extrapolation):")
for t in [15.0, 15.5, 16.0, 20.0]
    h = spline_ext(t)
    # Cumhaz beyond boundary = cumhaz(15) + h(15) * (t - 15)
    H_beyond = cumhaz_spline(15.0) + 0.18 * (t - 15.0)
    println("  t=$t: h(t) = $(round(h, digits=4)), H(0,t) ≈ $(round(H_beyond, digits=4))")
end
println()

# The key question: is the INTEGRATED spline being evaluated correctly?
# Let's check the spline basis functions' integrals
println("=" ^ 60)
println("Integrated basis function analysis")
println("=" ^ 60)
println()

# Create integral of each basis function
# This is what contributes to cumulative hazard
for j in 1:length(B)
    # Create spline with only coef j = 1
    test_coefs = zeros(length(B))
    test_coefs[j] = 1.0
    test_spline = Spline(B, test_coefs)
    test_cumhaz = integral(test_spline)
    
    println("Integrated B_$j at panel times:")
    for t in [0.0, 5.0, 10.0, 15.0]
        val = test_cumhaz(t)
        println("  ∫₀^$t B_$j(s)ds = $(round(val, digits=4))")
    end
    println()
end

println("=" ^ 60)
println("CONCLUSION")
println("=" ^ 60)
println("""
The integrated basis functions should satisfy partition of unity:
  Σⱼ ∫₀^t Bⱼ(s)ds = t  (for all t in [0, 15])

Let's verify this...
""")

# Sum of all integrated basis functions
total_integral_check = zeros(length(B))
for t in [5.0, 10.0, 15.0]
    total = 0.0
    for j in 1:length(B)
        test_coefs = zeros(length(B))
        test_coefs[j] = 1.0
        test_spline = Spline(B, test_coefs)
        test_cumhaz = integral(test_spline)
        total += test_cumhaz(t)
    end
    println("  Σⱼ ∫₀^$t Bⱼ(s)ds = $(round(total, digits=4)) (should be $t)")
end
