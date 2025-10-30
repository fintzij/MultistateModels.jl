using DiffResults, ForwardDiff, Optimization, OptimizationOptimJL, Symbolics

# objective function
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 1.0]

# set equality constraint x[1]=x[2]
cons(res, x, p) = (res .= [x[1] - x[2]])
lc = [0.0,]
uc = [0.0,]

# set up problem
optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(), cons = cons)
prob = OptimizationProblem(optprob, x0, _p, lcons = lc, ucons = uc)

# solve
sol = solve(prob, IPNewton())

# try to get hessian
diffres = DiffResults.HessianResult(sol.u)
obj = x -> rosenbrock(x, _p)
diffres = ForwardDiff.hessian!(diffres, obj, sol.u)

DiffResults.hessian(diffres) #this should not be dense


# with Symbolics
using Symbolics

@variables x1 x2
s = (1 - x1)^2 + (x1 - x1^2)^2

Symbolics.jacobian(Symbolics.gradient(s, [x1, x2]), [x1, x2])