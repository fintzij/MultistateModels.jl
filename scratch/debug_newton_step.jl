# Debug script to trace Newton step directions in PIJCV
# This will show exactly what's happening with the gradient and Hessian

using Pkg
Pkg.activate(".")

# Enable debug logging for MultistateModels
ENV["JULIA_DEBUG"] = "MultistateModels"

using MultistateModels
using DataFrames
using Random
using Printf
using Logging

# Set up debug logging to see all messages
global_logger(ConsoleLogger(stderr, Logging.Debug))

println("="^80)
println("DEBUG: Newton Step Trace for PIJCV")
println("="^80)

# Small test case
n = 30
max_time = 5.0
Random.seed!(12345)

# True hazards (semi-Markov with sojourn-dependent h23)
true_h12(t) = 0.3 * sqrt(max(t, 0.001))
true_h13(t) = 0.1 + 0.02 * t
true_h23(s) = 0.4 * exp(-0.1 * s)

true_H12(t) = 0.3 * (2/3) * t^1.5
true_H13(t) = 0.1 * t + 0.01 * t^2
true_H23(s) = 4.0 * (1 - exp(-0.1 * s))

function find_event_time(t_max, u, H_fn)
    target = -log(u)
    t_lo, t_hi = 0.0, t_max
    for _ in 1:100
        t_mid = (t_lo + t_hi) / 2
        if H_fn(t_mid) < target
            t_lo = t_mid
        else
            t_hi = t_mid
        end
        abs(t_hi - t_lo) < 1e-6 && break
    end
    return (t_lo + t_hi) / 2
end

function simulate_subject(id, max_time; rng)
    records = NamedTuple[]
    current_state = 1
    current_time = 0.0
    
    while current_state != 3 && current_time < max_time
        if current_state == 1
            u = rand(rng)
            t_event = find_event_time(max_time, u, t -> true_H12(t) + true_H13(t))
            
            if t_event >= max_time
                push!(records, (id=id, tstart=current_time, tstop=max_time, statefrom=1, stateto=1, obstype=2))
                break
            end
            
            h12_t = true_h12(t_event)
            h13_t = true_h13(t_event)
            prob_12 = h12_t / (h12_t + h13_t)
            
            if rand(rng) < prob_12
                push!(records, (id=id, tstart=current_time, tstop=t_event, statefrom=1, stateto=2, obstype=1))
                current_state = 2
            else
                push!(records, (id=id, tstart=current_time, tstop=t_event, statefrom=1, stateto=3, obstype=1))
                current_state = 3
            end
            current_time = t_event
            
        elseif current_state == 2
            u = rand(rng)
            sojourn = find_event_time(max_time - current_time, u, s -> true_H23(s))
            t_event = current_time + sojourn
            
            if t_event >= max_time
                push!(records, (id=id, tstart=current_time, tstop=max_time, statefrom=2, stateto=2, obstype=2))
                break
            end
            
            push!(records, (id=id, tstart=current_time, tstop=t_event, statefrom=2, stateto=3, obstype=1))
            current_state = 3
            current_time = t_event
        end
    end
    
    return records
end

all_records = NamedTuple[]
rng = Random.MersenneTwister(12345)
for id in 1:n
    append!(all_records, simulate_subject(id, max_time; rng=rng))
end
data = DataFrame(all_records)

n_12 = count(r -> r.statefrom == 1 && r.stateto == 2 && r.obstype == 1, eachrow(data))
n_13 = count(r -> r.statefrom == 1 && r.stateto == 3 && r.obstype == 1, eachrow(data))
n_23 = count(r -> r.statefrom == 2 && r.stateto == 3 && r.obstype == 1, eachrow(data))
println("\nData summary: N=$n, 1→2=$n_12, 1→3=$n_13, 2→3=$n_23")

# Create model
interior_knots = [1.0, 2.0, 3.0, 4.0]

println("\n" * "="^80)
println("Step 1: Computing V(λ) at several λ values manually")
println("="^80)

# First, let's manually compute V(λ) and its derivatives at a few λ values
# to understand the landscape

model = multistatemodel(
    Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true);
    data=data
)

# Fit with a few different λ values and check results
test_lambdas = [1e-4, 1e-2, 1.0, 100.0, 1e4, 1e6, 1e8]

println("\nλ\t\tloglik\t\th23@1.0/true")
println("-"^50)

for λ_val in test_lambdas
    model_test = multistatemodel(
        Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
        Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
        Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true);
        data=data
    )
    
    fitted = fit(model_test; lambda=λ_val, select_lambda=:none, verbose=false)
    
    pars = MultistateModels.get_parameters(fitted)
    h23_haz = fitted.hazards[fitted.hazkeys[:h23]]
    h_fit = h23_haz(1.0, pars.h23, NamedTuple())
    h_true = true_h23(1.0)
    
    ll = fitted.loglik.loglik
    
    println(@sprintf("%.0e\t\t%.2f\t\t%.4f", λ_val, ll, h_fit/h_true))
end

println("\n" * "="^80)
println("Step 2: Running PIJCV with debug logging enabled")
println("="^80)
println("Watch for 'PIJCV Newton step trace' and 'Newton step inputs' messages...")
println()

# Run PIJCV - the debug messages should appear
model_pijcv = multistatemodel(
    Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true);
    data=data
)

fitted_pijcv = fit(model_pijcv; verbose=true)

println("\n" * "="^80)
println("Results")
println("="^80)

# Show what we got
if hasproperty(fitted_pijcv, :smoothing_parameters)
    println("Selected λ: ", fitted_pijcv.smoothing_parameters)
end

# Check h23 ratios
pars_pijcv = MultistateModels.get_parameters(fitted_pijcv)
h23_pijcv = fitted_pijcv.hazards[fitted_pijcv.hazkeys[:h23]]

println("\nh23 fit/true ratios:")
for s in [0.5, 1.0, 2.0, 4.0]
    h_fit = h23_pijcv(s, pars_pijcv.h23, NamedTuple())
    h_true = true_h23(s)
    println("  s=$s: ratio = $(round(h_fit/h_true, digits=4))")
end
