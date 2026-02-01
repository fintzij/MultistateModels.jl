# Trace PIJCV criterion V(λ) at different λ values
# to understand why performance iteration pushes λ to maximum
using MultistateModels
using DataFrames
using Random
using Printf

println("=== PIJCV Criterion V(λ) Trace ===")

n = 50
max_time = 5.0
Random.seed!(12345)

# True hazards
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
println("Data: N=$n, 1→2=$n_12, 1→3=$n_13, 2→3=$n_23")

# Create model with splines
interior_knots = [1.0, 2.0, 3.0, 4.0]

println("\n=== Computing V(λ) at different λ values ===")

# Use a grid of λ values
lambda_grid = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7, 1e8]

results = []
for λ_val in lambda_grid
    println("\nTrying λ = $λ_val...")
    
    # Fit with this λ
    try
        # Reset model
        model_test = multistatemodel(
            Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
            Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
            Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true);
            data=data
        )
        
        fitted = fit(model_test; lambda=λ_val, select_lambda=:none, verbose=false)
        
        # Evaluate hazards
        pars = MultistateModels.get_parameters(fitted)
        h23_haz = fitted.hazards[fitted.hazkeys[:h23]]
        
        h23_ratios = Float64[]
        for s in [0.5, 1.0, 2.0, 4.0]
            h_true = true_h23(s)
            h_fit = h23_haz(s, pars.h23, NamedTuple())
            push!(h23_ratios, h_fit / h_true)
        end
        
        # Get EDF if available
        edf = hasproperty(fitted, :effective_df) ? fitted.effective_df : NaN
        
        push!(results, (lambda=λ_val, h23_ratios=h23_ratios, edf=edf))
        println("  h23 ratios: ", round.(h23_ratios, digits=3))
        
    catch e
        println("  ERROR: ", e)
        println("  ", sprint(showerror, e))
    end
end

println("\n=== Summary ===")
println("λ\t\tEDF\th23@0.5\th23@1.0\th23@2.0\th23@4.0")
println("-"^70)
for r in results
    λ_str = @sprintf("%.0e", r.lambda)
    edf_str = isnan(r.edf) ? "N/A" : @sprintf("%.1f", r.edf)
    ratios_str = join([@sprintf("%.3f", x) for x in r.h23_ratios], "\t")
    println("$λ_str\t\t$edf_str\t$ratios_str")
end

# Now run PIJCV to see what happens
println("\n\n=== Running PIJCV (default) ===")

model_pijcv = multistatemodel(
    Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true);
    data=data
)

# Run with verbose
fitted_pijcv = fit(model_pijcv; verbose=true)

# Check selected λ
println("\n=== Selected Lambda ===")
if hasproperty(fitted_pijcv, :selected_lambda)
    println("Selected lambda: ", fitted_pijcv.selected_lambda)
end

if hasproperty(fitted_pijcv, :effective_df)
    println("Effective df: ", fitted_pijcv.effective_df)
end

# Evaluate h23
pars_pijcv = MultistateModels.get_parameters(fitted_pijcv)
h23_pijcv = fitted_pijcv.hazards[fitted_pijcv.hazkeys[:h23]]

println("\n--- PIJCV h23 Comparison ---")
for s in [0.5, 1.0, 2.0, 4.0]
    h_true = true_h23(s)
    h_fit = h23_pijcv(s, pars_pijcv.h23, NamedTuple())
    ratio = h_fit / h_true
    println("  s=$s: true=$(round(h_true, digits=3)), fitted=$(round(h_fit, digits=6)), ratio=$(round(ratio, digits=4))")
end
