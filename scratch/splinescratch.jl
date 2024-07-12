using BenchmarkTools
using BSplineKit
using Distributions
using LinearAlgebra
using Plots
using Random

# set up basis
# x = sort(rand(Uniform(0.25, 0.75), 20))
# breaks = sort(rand(Uniform(0.3, 0.7), 2))
x = [0.1, 0.5, 0.9]
z = collect(0.1:0.01:0.9)

function makebasis(Bb)
    M = zeros(maximum(map(x -> x[1],Bb)), length(Bb))
    n = length(Bb[1][2])
    for k in 1:length(Bb)
        M[range(Bb[k][1], step = -1, length = n), k] = [Bb[k][2]...]
    end
    return transpose(M)
end

# generate the basis object
B = BSplineBasis(BSplineOrder(4), copy(x))
Bb = B.(z)
bsp_basis = makebasis(Bb)


# so for the baseline hazard
# Steps 1 and 2 only need to be done once
# 1. generate BSpline basis (B),  
# 2. generate recombined basis (BN) and get recombination matrix (RM).
# The following happens for each new value of recombined coefficients
# 3. B-spline coefficients = RM * recombined coefficients


# example
x = [0.1, 0.3, 0.45, 0.6, 0.9]
B = BSplineBasis(BSplineOrder(4), copy(x))
R = RecombinedBSplineBasis(B, Derivative(2))
M = R.M      

Bs = SplineExtrapolation(Spline(undef, B), Linear())

coefs_R = rand(length(R))
coefs_B = M * coefs_R

copyto!(Bs.spline.coefs, coefs_B)

# plot
z1 = collect(0.1:0.01:0.9)
z2 = collect(-0.1:0.01:0.1)
z3 = collect(0.9:0.01:1.1)

p = plot(z1, Bs.(z1))
plot!(p, z2, Bs.(z2); linestyle = :dash, color = :red)
plot!(p, z3, Bs.(z3); linestyle = :dash, color = :red)

# how to make integral?
Bi = integral(Bs)
Ri = SplineExtrapolation(Bi, Linear())

0.5 * 0.05 * (Bs(0.0) + Bs(0.05))
-(Ri(0.0) - Ri(0.05))

# notes
# the splines go in the spline hazard object
# spline extrapolations have a coefs field
# need to compute the risk period where the hazard is nonzero when setting parameters