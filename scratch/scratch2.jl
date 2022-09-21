using LinearAlgebra
using DifferentialEquations

# example of matrix exponential
# stochastic rate matrix
Q = [-2.0 1.0 1.0; 
     1.2 -3.1 1.9;
     0      0   0]

# compute matrix exponential by hand
S = eigen(Q)
Sexp = S.vectors * diagm(exp.(S.values)) * inv(S.vectors)

Q*S.vectors ≈ S.vectors*diagm(S.values)

# what if we want to change the time interval? e.g., t2 - t1 = 5.4
Sexp = S.vectors * diagm(exp.(S.values .* 5.4)) * inv(S.vectors)

# try solving numerically, time homogeneous case
function update_func(du, u, p, t)
     # kolmogorov forward equation
     du = u * p
end

# try solving numerically - time inhomogeneous case
# function update_func(du, u, p, t)
#      # kolmogorov forward equation
#      Q = make_Qmat(p,t) # because Q needs to be updated at each time step, potentially allocating and slow
#      du = u * Q
# end

A = DiffEqArrayOperator(diagm(ones(3)), update_func = update_func)
prob = ODEProblem(A, diagm(ones(3)), (10, 50.), p)
solve(prob)