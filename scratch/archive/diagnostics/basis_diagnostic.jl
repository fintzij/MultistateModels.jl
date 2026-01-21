using BSplineKit

# Replicate exact setup from MultistateModels spline builder
degree = 3
intknots = [5.0, 10.0]
bknots = [0.0, 15.0]
allknots = unique(sort([bknots[1]; intknots; bknots[2]]))

# Build basis with Derivative(1) boundary conditions
B_orig = BSplineBasis(BSplineOrder(degree + 1), copy(allknots))
B = RecombinedBSplineBasis(B_orig, Derivative(1))

println("Allknots: ", allknots)
println("Number of original basis functions: ", length(B_orig))
println("Number of recombined basis functions: ", length(B))
println()

# Evaluate all basis functions at key points
println("=== Basis function values ===")
times = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
for t in times
    vals = [B[i](t) for i in 1:length(B)]
    println("t=$t: B=$(round.(vals, digits=4))")
end
println()

# Key insight: create sample spline with known coefficients
coefs = [0.08, 0.10, 0.14, 0.18]  # True h12 coefficients
spline = Spline(B, coefs)
spline_ext = SplineExtrapolation(spline, BSplineKit.SplineExtrapolations.Flat())

println("=== Spline hazard at observation times (true coefs) ===")
for t in times
    h = spline_ext(t)
    println("t=$t: h(t) = $(round(h, digits=6))")
end
println()

# Critical: hazard at boundary
println("=== Hazard at/near boundary ===")
for t in [14.0, 14.5, 14.9, 14.99, 15.0]
    h = spline_ext(t)
    println("t=$t: h(t) = $(round(h, digits=6))")
end
println()

# What if coef_4 is much larger?
bad_coefs = [0.08, 0.10, 0.14, 0.867]  # Estimated h12 coefs
bad_spline = Spline(B, bad_coefs)
bad_ext = SplineExtrapolation(bad_spline, BSplineKit.SplineExtrapolations.Flat())

println("=== Hazard with ESTIMATED coefs (coef_4 overestimated) ===")
for t in times
    h_true = spline_ext(t)
    h_bad = bad_ext(t)
    println("t=$t: h_true=$(round(h_true, digits=4)), h_bad=$(round(h_bad, digits=4)), ratio=$(round(h_bad/h_true, digits=2))")
end
