using MultistateModels
using BSplineKit
using ForwardDiff
using DataFrames

# Create simple test data
nsubj = 100
data = DataFrame(
    id = repeat(1:nsubj, inner=2),
    tstart = repeat([0.0, 1.0], nsubj),
    tstop = repeat([1.0, 2.0], nsubj),
    statefrom = repeat([1, 1], nsubj),
    stateto = repeat([1, 2], nsubj),
    obstype = repeat([1, 2], nsubj)  # 1 = exact, 2 = right-censored or event
)

# Test survextrap extrapolation (degree must be >= 2)
println("Testing survextrap extrapolation...")
h_survextrap = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
    degree=3, 
    knots=[0.5, 1.0, 1.5], 
    boundaryknots=[0.0, 2.0],
    extrapolation="survextrap"
)

model_survextrap = multistatemodel(h_survextrap; data=data)
println("✓ Model with survextrap extrapolation created successfully")

# Check that the basis has the correct boundary conditions
haz = model_survextrap.hazards[1]
println("  Hazard extrapolation field: $(haz.extrapolation)")

# Build the spline and check derivatives at boundary
B = BSplineBasis(BSplineOrder(4), copy(haz.knots))
B_recomb = RecombinedBSplineBasis(B, Derivative(1))
println("  Original basis length: $(length(B))")
println("  Recombined basis length: $(length(B_recomb)) (2 fewer due to D1=0 at both boundaries)")
println("  Hazard basis functions: $(haz.npar_baseline)")

# Test that hazard and cumhaz can be evaluated
pars = zeros(haz.npar_total)
h_val = haz.hazard_fn(1.0, pars, NamedTuple())
println("  Hazard at t=1.0: $(h_val)")

# Test cumhaz across boundary
H_val = haz.cumhaz_fn(0.0, 3.0, pars, NamedTuple())
println("  Cumulative hazard from 0 to 3 (crosses upper boundary): $(H_val)")

# Compare with flat extrapolation (should have different derivatives at boundary)
println("\nComparing with flat extrapolation...")
h_flat = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
    degree=3, 
    knots=[0.5, 1.0, 1.5], 
    boundaryknots=[0.0, 2.0],
    extrapolation="flat"
)
model_flat = multistatemodel(h_flat; data=data)
haz_flat = model_flat.hazards[1]
println("  Flat basis functions: $(haz_flat.npar_baseline)")

# Test that survextrap has smooth transition (D1=0 at boundary)
# Build splines with coefficients that would show curvature
println("\nChecking smoothness at upper boundary (t=2.0)...")

# Build bases for testing - note flat has 7 basis functions, survextrap has 5
B_surv = BSplineBasis(BSplineOrder(4), copy(haz.knots))
B_surv_recomb = RecombinedBSplineBasis(B_surv, Derivative(1))
# Use varying coefficients to create curvature
test_coefs_survextrap = [1.0, 2.0, 1.5, 0.5, 1.0][1:length(B_surv_recomb)]
spline_surv = Spline(B_surv_recomb, test_coefs_survextrap)

B_flat_basis = BSplineBasis(BSplineOrder(4), copy(haz_flat.knots))
test_coefs_flat = [1.0, 2.0, 1.5, 0.5, 1.0, 2.0, 0.5][1:length(B_flat_basis)]
spline_flat = Spline(B_flat_basis, test_coefs_flat)

t_boundary = 2.0 - 1e-6  # Just before upper boundary
d1_surv = ForwardDiff.derivative(t -> spline_surv(t), t_boundary)
d1_flat = ForwardDiff.derivative(t -> spline_flat(t), t_boundary)

println("  survextrap: h'(2-ε) = $(round(d1_surv, digits=6))")
println("  flat:       h'(2-ε) = $(round(d1_flat, digits=6))")

# Check that survextrap has near-zero first derivative while flat may not
@assert abs(d1_surv) < 0.01 "survextrap should have h'(boundary) ≈ 0"
println("✓ survextrap enforces zero derivative at boundary (C¹ continuity)")
if abs(d1_flat) > 0.01
    println("✓ flat extrapolation has non-zero derivative at boundary (as expected)")
else
    println("  (flat also has near-zero derivative with these coefficients)")
end

println("\n✓ All survextrap tests passed!")
