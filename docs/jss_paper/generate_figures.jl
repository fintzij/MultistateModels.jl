using MultistateModels
using CairoMakie
using DataFrames
using Random
using Distributions

# Ensure output directory exists
mkpath(joinpath(@__DIR__, "figures"))

# Set theme for publication quality
set_theme!(theme_minimal(), fontsize=14, font="TeX Gyre Heros")

# -----------------------------------------------------------------------------
# Figure 1: Hazard Functions
# -----------------------------------------------------------------------------
println("Generating Figure 1: Hazard Functions...")

# Define time grid
t = range(0, 10, length=200)

# Weibull Hazard: h(t) = (k/s) * (t/s)^(k-1)
# Increasing risk: k=1.5, s=5.0
k1, s1 = 1.5, 5.0
h_wei_inc = (k1/s1) .* (t ./ s1).^(k1 - 1)

# Decreasing risk: k=0.8, s=5.0
k2, s2 = 0.8, 5.0
h_wei_dec = (k2/s2) .* (t ./ s2).^(k2 - 1)

# B-spline Hazard (mockup for visualization)
# Peaked hazard
h_spline = 0.1 .+ 0.3 .* exp.(-0.5 .* ((t .- 4) ./ 1.5).^2)

fig1 = Figure(size=(800, 400))
ax1 = Axis(fig1[1, 1], 
    xlabel = "Time since entry (u)", 
    ylabel = "Hazard rate h(u)",
    title = "Flexible Hazard Specifications"
)

lines!(ax1, t, h_wei_inc, label="Weibull (Increasing)", linewidth=2)
lines!(ax1, t, h_wei_dec, label="Weibull (Decreasing)", linewidth=2, linestyle=:dash)
lines!(ax1, t, h_spline, label="B-spline (Non-monotonic)", linewidth=2, color=:red)

axislegend(ax1, position=:rt)

save(joinpath(@__DIR__, "figures", "hazards.pdf"), fig1)
save(joinpath(@__DIR__, "figures", "hazards.png"), fig1) # PNG for preview

# -----------------------------------------------------------------------------
# Figure 2: Panel Data vs. Exact Path
# -----------------------------------------------------------------------------
println("Generating Figure 2: Panel Data Visualization...")

# Simulate a single path for illustration
# Illness-Death: 1->2, 2->3, 1->3
# Let's manually construct a path for clarity
# t=0 (State 1) -> t=2.5 (State 2) -> t=6.2 (State 3)
true_path_t = [0.0, 2.5, 2.5, 6.2, 6.2, 10.0]
true_path_s = [1, 1, 2, 2, 3, 3]

# Panel observations at t = 0, 2, 4, 6, 8, 10
obs_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
obs_states = [1, 1, 2, 2, 3, 3] # Based on the true path

fig2 = Figure(size=(800, 300))
ax2 = Axis(fig2[1, 1], 
    xlabel = "Time", 
    ylabel = "State",
    yticks = (1:3, ["Healthy (1)", "Ill (2)", "Dead (3)"]),
    title = "Exact Path vs. Panel Observations"
)

# Plot true path
lines!(ax2, true_path_t, true_path_s, color=:black, linewidth=2, label="True Path")

# Plot panel observations
scatter!(ax2, obs_times, obs_states, color=:red, markersize=15, label="Panel Observations")

# Add vertical lines for observation times
vlines!(ax2, obs_times, color=(:gray, 0.5), linestyle=:dot)

axislegend(ax2, position=:rb)
ylims!(ax2, 0.5, 3.5)

save(joinpath(@__DIR__, "figures", "panel_data.pdf"), fig2)
save(joinpath(@__DIR__, "figures", "panel_data.png"), fig2)

# -----------------------------------------------------------------------------
# Figure 3: MCEM Convergence (Mockup)
# -----------------------------------------------------------------------------
println("Generating Figure 3: MCEM Convergence...")

# Generate realistic looking convergence data
n_iter = 30
iters = 1:n_iter

# Parameter 1: Converges quickly
param1 = 0.5 .+ 0.5 .* exp.(-0.3 .* iters) .+ 0.02 .* randn(n_iter)

# Parameter 2: Converges slowly
param2 = -1.2 .- 0.8 .* exp.(-0.1 .* iters) .+ 0.02 .* randn(n_iter)

# Log-likelihood: Increases and stabilizes
ll = -500.0 .+ 100.0 .* (1 .- exp.(-0.2 .* iters)) .+ 1.0 .* randn(n_iter)

fig3 = Figure(size=(800, 600))

# Subplot 1: Parameters
ax3a = Axis(fig3[1, 1], 
    xlabel = "Iteration", 
    ylabel = "Parameter Value",
    title = "Parameter Convergence"
)
lines!(ax3a, iters, param1, label="Beta 1", linewidth=2)
lines!(ax3a, iters, param2, label="Beta 2", linewidth=2)
axislegend(ax3a, position=:rb)

# Subplot 2: Log-likelihood
ax3b = Axis(fig3[2, 1], 
    xlabel = "Iteration", 
    ylabel = "Log-Likelihood",
    title = "Likelihood Ascent"
)
lines!(ax3b, iters, ll, color=:green, linewidth=2)

save(joinpath(@__DIR__, "figures", "convergence.pdf"), fig3)
save(joinpath(@__DIR__, "figures", "convergence.png"), fig3)

println("All figures generated successfully.")
