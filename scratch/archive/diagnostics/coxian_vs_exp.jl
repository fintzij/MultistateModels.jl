using LinearAlgebra

# Create a 3-phase Coxian that matches exponential with rate lambda = 0.247
lambda_rate = 0.247

n_phases = 3
Q = zeros(n_phases + 1, n_phases + 1)

uniform_rate = lambda_rate
for i in 1:n_phases
    if i < n_phases
        Q[i, i+1] = uniform_rate
        Q[i, n_phases+1] = uniform_rate
        Q[i, i] = -2 * uniform_rate
    else
        Q[i, n_phases+1] = uniform_rate
        Q[i, i] = -uniform_rate
    end
end

println("Q matrix for Coxian-3:")
display(Q)

pi0 = [1.0, 0.0, 0.0, 0.0]

t = 9.991
P = exp(Q * t)
println("\nP(t) = exp(Qt):")
display(round.(P, digits=6))

survival_coxian = sum(pi0' * P .* [1.0, 1.0, 1.0, 0.0])
println("\nCoxian survival P(T > t) = ", survival_coxian)

survival_exp = exp(-lambda_rate * t)
println("Exponential survival P(T > t) = ", survival_exp)

println("\nDifference: ", survival_coxian - survival_exp)
println("Ratio: ", survival_coxian / survival_exp)
