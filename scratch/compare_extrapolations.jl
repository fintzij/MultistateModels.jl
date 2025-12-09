using BSplineKit
using BSplineKit.SplineExtrapolations: Flat, Linear, SplineExtrapolation
using ForwardDiff
using CairoMakie

# Set up the spline basis and coefficients
knots = [0.0, 0.5, 1.0, 1.5, 2.0]
order = 4  # degree 3
B_standard = BSplineBasis(BSplineOrder(order), knots)
B_neumann = RecombinedBSplineBasis(B_standard, Derivative(1))  # h'=0 at boundaries (constant extrapolation)

println("Standard basis (for linear): $(length(B_standard)) functions")
println("Neumann basis (for constant): $(length(B_neumann)) functions")

# Use coefficients that create interesting curvature near boundaries
# Standard basis has 7 coefficients, Neumann has 5
coefs_standard = [0.5, 1.5, 2.5, 1.0, 0.8, 1.2, 2.0]
coefs_neumann = [1.0, 2.0, 1.2, 0.7, 1.5]

# Build splines
spline_standard = Spline(B_standard, coefs_standard)
spline_neumann = Spline(B_neumann, coefs_neumann)

# Create extrapolated versions for the two supported methods
spline_linear = SplineExtrapolation(spline_standard, Linear())
spline_constant = SplineExtrapolation(spline_neumann, Flat())

# Evaluation range (extends beyond knot boundaries)
t_range = range(-0.5, 3.0, length=500)
t_lo, t_hi = knots[1], knots[end]

# Evaluate hazards
h_linear = [spline_linear(t) for t in t_range]
h_constant = [spline_constant(t) for t in t_range]

# Evaluate derivatives at boundary (for annotation)
d1_linear_hi = ForwardDiff.derivative(t -> spline_standard(t), t_hi - 1e-6)
d1_constant_hi = ForwardDiff.derivative(t -> spline_neumann(t), t_hi - 1e-6)

println("\nDerivatives at upper boundary (t=2.0):")
println("  linear:   h'(2⁻) = $(round(d1_linear_hi, digits=4))")  
println("  constant: h'(2⁻) = $(round(d1_constant_hi, digits=4))")

# Create figure
fig = Figure(size=(900, 600))

# Main plot
ax = Axis(fig[1, 1],
    xlabel = "Time t",
    ylabel = "Hazard h(t)",
    title = "Spline Hazard Extrapolation Methods Comparison"
)

# Shade extrapolation regions
vspan!(ax, -0.5, t_lo, color=(:gray, 0.1))
vspan!(ax, t_hi, 3.0, color=(:gray, 0.1))

# Add vertical lines at boundaries
vlines!(ax, [t_lo, t_hi], color=:gray, linestyle=:dash, linewidth=1, label="Knot boundaries")

# Plot hazards - only the two supported options now
lines!(ax, collect(t_range), h_linear, color=:red, linewidth=2, label="linear (continues with slope)")
lines!(ax, collect(t_range), h_constant, color=:green, linewidth=2, label="constant (smooth, h'=0 at boundary)")

# Add legend
axislegend(ax, position=:lt)

# Add annotation
text!(ax, 2.3, 2.5, text="Extrapolation\nregion", fontsize=12, color=:gray)
text!(ax, -0.3, 2.5, text="Extrapolation\nregion", fontsize=12, color=:gray)

# Derivative comparison subplot
ax2 = Axis(fig[2, 1],
    xlabel = "Time t", 
    ylabel = "Derivative h'(t)",
    title = "Hazard Derivatives (showing continuity at boundaries)"
)

# Compute derivatives
h_linear_deriv = [ForwardDiff.derivative(t -> spline_linear(t), t) for t in t_range]
h_constant_deriv = [ForwardDiff.derivative(t -> t < t_lo || t > t_hi ? spline_constant(t) : spline_neumann(t), t) for t in t_range]

vspan!(ax2, -0.5, t_lo, color=(:gray, 0.1))
vspan!(ax2, t_hi, 3.0, color=(:gray, 0.1))
vlines!(ax2, [t_lo, t_hi], color=:gray, linestyle=:dash, linewidth=1)
hlines!(ax2, [0], color=:black, linestyle=:dot, linewidth=0.5)

lines!(ax2, collect(t_range), h_linear_deriv, color=:red, linewidth=2, label="linear")
lines!(ax2, collect(t_range), h_constant_deriv, color=:green, linewidth=2, label="constant")

axislegend(ax2, position=:rt)

# Save figure
save("scratch/extrapolation_comparison.png", fig, px_per_unit=2)
println("\nFigure saved to scratch/extrapolation_comparison.png")

fig
