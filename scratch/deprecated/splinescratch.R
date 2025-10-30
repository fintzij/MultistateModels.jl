library(splines2) # M-splines, also has natural splines and B splines 

# need to establish that the B-spline basis is the same as BSplineKit.jl
x = seq(0.1, 0.9, by = 0.1)
breaks = 0.5
B = bSpline(x, knots = breaks, degree = 2)

# try natural spline from splines2
x1 = sort(runif(50, 0.2, 0.7))
x2 = sort(runif(10, 0, 0.2))
x3 = sort(runif(10, 0.7, 1))
knots = 0.45
natsp = naturalSpline(x1, knots = knots, intercept = TRUE)
msp <- mSpline(x1, knots = knots, degree = 2, intercept = TRUE)

# outside the boundary
natsp2 = rbind(predict(update(natsp, x = x2)), natsp, predict(update(natsp, x = x3)))
# msp2 = rbind(msp, predict(update(msp, x = x2)))

# let's see how the basis is constructed
(natsp[50,2] - natsp[49,2]) 
(natsp2[51,2] - natsp2[50,2])

# coefs
beta <- seq.int(0.2, 0.00, length.out = ncol(natsp))

plot(seq(0,2,by=delta), natsp2 %*% beta, "l")
plot(seq(0,2,by=delta), msp2 %*% c(1.0, beta), "l")  # can also be negative when extrapolating

# looks like the natural spline basis should continue the last value in the last column and slope based on the second to last column past the right endpoint. the reverse before the left endpoint. let's see for the mspline
