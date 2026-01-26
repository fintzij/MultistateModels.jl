using MultistateModels
using Random
using DataFrames

# Runtime comparison
n = 1000
true_shape = 1.5
true_rate = 0.3
max_time = 5.0
nknots = 5

Random.seed!(12345)

E = -log.(rand(n))
event_times = (E ./ true_rate) .^ (1 / true_shape)
obs_times = min.(event_times, max_time)
status = Int.(event_times .<= max_time)

surv_data = DataFrame(
    id = 1:n, tstart = zeros(n), tstop = obs_times,
    statefrom = ones(Int, n),
    stateto = ifelse.(status .== 1, 2, 1),
    obstype = ones(Int, n)
)

h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
             degree = 3, knots = [max_time/2],
             boundaryknots = [0.0, max_time],
             natural_spline = true, monotone = 1)
model = multistatemodel(h12; data=surv_data)
calibrate_splines!(model; nknots=nknots, verbose=false)

println("\n=== Runtime Comparison ===")

# Warmup
fit(model; penalty=:auto, select_lambda=:pijcv, vcov_type=:none, verbose=false)

t_pijcv = @elapsed fit(model; penalty=:auto, select_lambda=:pijcv, vcov_type=:none, verbose=false)
println("PIJCV:   ", round(t_pijcv, digits=3), "s")

t_cv10 = @elapsed fit(model; penalty=:auto, select_lambda=:cv10, vcov_type=:none, verbose=false)
println("CV10:    ", round(t_cv10, digits=3), "s")

t_efs = @elapsed fit(model; penalty=:auto, select_lambda=:efs, vcov_type=:none, verbose=false)
println("EFS:     ", round(t_efs, digits=3), "s")

println("\nRatio CV10/PIJCV: ", round(t_cv10/t_pijcv, digits=2), "x")
