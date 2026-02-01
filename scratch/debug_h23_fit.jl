# Debug script for h23 fitting issue
# Trace likelihood contributions to understand the bug
using MultistateModels
using DataFrames
using Random
using Statistics

println("=== Detailed Likelihood Trace ===")

n = 50  # Smaller for easier debugging
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

# Extract paths
import MultistateModels: extract_paths, SamplePath
paths = extract_paths(data)

# Look at sojourn times for state 2→3 transitions
function show_first_23_transition(paths)
    for path in paths
        for i in 1:(length(path.times)-1)
            if path.states[i] == 2 && path.states[i+1] == 3
                sojourn = path.times[i+1] - path.times[i]
                println("  Subject $(path.subj): entered t=$(round(path.times[i], digits=2)), exited t=$(round(path.times[i+1], digits=2)), sojourn=$(round(sojourn, digits=2))")
                return
            end
        end
    end
end

println("\nFirst 2→3 transition:")
show_first_23_transition(paths)

# Now compute manual likelihood contribution for a subject that went 1→2→3
println("\n=== Manual Likelihood for Subject with 1→2→3 Path ===")

# Find first subject with 1→2→3 path
function find_123_subject(paths)
    for path in paths
        if length(path.states) >= 3 && path.states[1] == 1 && path.states[2] == 2 && path.states[3] == 3
            return path
        end
    end
    return nothing
end

path123 = find_123_subject(paths)
if path123 === nothing
    println("No 1→2→3 paths found!")
else
    subj_data = data[data.id .== path123.subj, :]
    println("Data rows:")
    println(subj_data)
    
    println("\nSamplePath: times=$(path123.times), states=$(path123.states)")

# Manually compute log-likelihood contribution
function manual_loglik(path, h12, h13, h23, H12, H13, H23)
    ll = 0.0
    sojourn = 0.0
    
    for i in 1:(length(path.times)-1)
        increment = path.times[i+1] - path.times[i]
        lb = sojourn
        ub = sojourn + increment
        sfrom = path.states[i]
        sto = path.states[i+1]
        
        println("\nInterval $i: clock=[$(round(path.times[i], digits=2)), $(round(path.times[i+1], digits=2))], state=$sfrom→$sto")
        println("  sojourn: lb=$(round(lb, digits=3)), ub=$(round(ub, digits=3))")
        
        if sfrom == 1
            # Survival in state 1: subtract cumulative hazards h12 and h13
            cumhaz_12 = H12(ub) - H12(lb)
            cumhaz_13 = H13(ub) - H13(lb)
            ll -= cumhaz_12 + cumhaz_13
            println("  -cumhaz h12([$(round(lb, digits=3)), $(round(ub, digits=3))]) = -$(round(cumhaz_12, digits=3))")
            println("  -cumhaz h13([$(round(lb, digits=3)), $(round(ub, digits=3))]) = -$(round(cumhaz_13, digits=3))")
            
            if sfrom != sto
                if sto == 2
                    ll += log(h12(ub))
                    println("  +log(h12($(round(ub, digits=3)))) = $(round(log(h12(ub)), digits=3))")
                elseif sto == 3
                    ll += log(h13(ub))
                    println("  +log(h13($(round(ub, digits=3)))) = $(round(log(h13(ub)), digits=3))")
                end
            end
        elseif sfrom == 2
            # Survival in state 2: subtract cumulative hazard h23 (sojourn time!)
            cumhaz_23 = H23(ub) - H23(lb)
            ll -= cumhaz_23
            println("  -cumhaz h23([$(round(lb, digits=3)), $(round(ub, digits=3))]) = -$(round(cumhaz_23, digits=3))")
            
            if sfrom != sto
                ll += log(h23(ub))
                println("  +log(h23($(round(ub, digits=3)))) = $(round(log(h23(ub)), digits=3))")
            end
        end
        
        # Update sojourn
        sojourn = (sfrom != sto) ? 0.0 : ub
        println("  → sojourn reset: $(sfrom != sto ? "YES (to 0.0)" : "NO (=$ub)")")
    end
    
    return ll
end

ll_manual = manual_loglik(path123, true_h12, true_h13, true_h23, true_H12, true_H13, true_H23)
println("\n\nManual log-likelihood: $(round(ll_manual, digits=3))")
end

# Now compute using MultistateModels
interior_knots = [1.0, 2.0, 3.0, 4.0]
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true)
h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true)
h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true)

model = multistatemodel(h12, h13, h23; data=data)

# Get the initial parameters and compute the likelihood
pars_init = MultistateModels.get_parameters(model)
println("\nInitial parameters h23: ", pars_init.h23)
println("h23(0.5) with initial params: ", model.hazards[3](0.5, pars_init.h23, NamedTuple()))
println("h23(2.0) with initial params: ", model.hazards[3](2.0, pars_init.h23, NamedTuple()))

# FIT THE MODEL
println("\n\n=== FITTING THE MODEL ===")

# First, check what the initial likelihood is
import MultistateModels: ExactData, extract_paths, loglik_exact
samplepaths = extract_paths(model)
data_exact = ExactData(model, samplepaths)

pars_init = model.parameters.flat
println("\n--- Initial State ---")
println("Initial parameters: $(length(pars_init)) parameters")
println("All params: ", pars_init)
println("All positive? ", all(pars_init .>= 0))

# Find negative parameters
neg_idx = findall(pars_init .< 0)
if !isempty(neg_idx)
    println("WARNING: Negative parameters at indices: $neg_idx")
    println("  Values: $(pars_init[neg_idx])")
end

ll_init = loglik_exact(pars_init, data_exact; neg=false)
println("\nInitial log-likelihood: $ll_init")

# Evaluate initial hazard values
h23_haz_init = model.hazards[model.hazkeys[:h23]]
pars_h23_init = MultistateModels.get_parameters(model).h23
println("\nInitial h23 hazard values:")
for s in [0.5, 1.0, 2.0, 4.0]
    h_true = true_h23(s)
    h_init = h23_haz_init(s, pars_h23_init, NamedTuple())
    ratio = h_init / h_true
    println("  s=$s: true=$(round(h_true, digits=3)), init=$(round(h_init, digits=3)), ratio=$(round(ratio, digits=2))")
end

# Check parameter bounds
println("\n--- Parameter Bounds ---")
lb = model.bounds.lb
ub = model.bounds.ub
println("Lower bounds: ", lb)
println("Upper bounds: ", ub)

# Fix initial parameters to be positive before fitting
println("\n--- Fixing Initial Parameters ---")
pars_fixed = max.(pars_init, 1e-9)  # Ensure all positive
import MultistateModels: set_parameters_flat!
set_parameters_flat!(model, pars_fixed)
println("Fixed parameters: ", model.parameters.flat)

ll_fixed = loglik_exact(model.parameters.flat, data_exact; neg=false)
println("Fixed log-likelihood: $ll_fixed")

# Now fit using the standard function with penalty=:none
println("\n--- Fitting with penalty=:none ---")
fitted_unpen = fit(model; penalty=:none, verbose=false)

# Get fitted parameters
pars_unpen = MultistateModels.get_parameters(fitted_unpen)
h23_haz_unpen = fitted_unpen.hazards[fitted_unpen.hazkeys[:h23]]

println("\n--- Unpenalized h23 Comparison ---")
for s in [0.5, 1.0, 2.0, 4.0]
    h_true = true_h23(s)
    h_fit = h23_haz_unpen(s, pars_unpen.h23, NamedTuple())
    ratio = h_fit / h_true
    println("  s=$s: true=$(round(h_true, digits=3)), fitted=$(round(h_fit, digits=3)), ratio=$(round(ratio, digits=2))")
end

# Now fit with PIJCV (default) and see what λ is selected
println("\n\n--- Fitting with PIJCV (default) ---")

# Rebuild model again
model3 = multistatemodel(
    Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true),
    Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots, boundaryknots=[0.0, max_time], natural_spline=true);
    data=data
)

# Fix negative initial parameters
pars_init3 = model3.parameters.flat
pars_fixed3 = max.(pars_init3, 1e-9)
set_parameters_flat!(model3, pars_fixed3)

fitted_pijcv = fit(model3; verbose=true)

# Get the selected lambda
println("\n--- PIJCV Selected Lambda ---")
if hasproperty(fitted_pijcv, :lambda)
    println("Selected λ: ", fitted_pijcv.lambda)
elseif hasproperty(fitted_pijcv, :smoothing_params)
    println("Smoothing params: ", fitted_pijcv.smoothing_params)
end

# Check what penalty is in the fitted model
if hasproperty(fitted_pijcv, :penalty)
    pen = fitted_pijcv.penalty
    println("Penalty type: ", typeof(pen))
    if hasproperty(pen, :terms)
        for (i, term) in enumerate(pen.terms)
            println("  Term $i lambda: $(term.lambda)")
        end
    end
end

# Evaluate fitted hazards
pars_pijcv = MultistateModels.get_parameters(fitted_pijcv)
h23_haz_pijcv = fitted_pijcv.hazards[fitted_pijcv.hazkeys[:h23]]

println("\n--- PIJCV h23 Comparison ---")
for s in [0.5, 1.0, 2.0, 4.0]
    h_true = true_h23(s)
    h_fit = h23_haz_pijcv(s, pars_pijcv.h23, NamedTuple())
    ratio = h_fit / h_true
    println("  s=$s: true=$(round(h_true, digits=3)), fitted=$(round(h_fit, digits=6)), ratio=$(round(ratio, digits=4))")
end

pars_pen = MultistateModels.get_parameters(fitted_unpen)
h23_haz_pen = fitted_unpen.hazards[fitted_unpen.hazkeys[:h23]]

println("--- Unpenalized h23 Comparison ---")
for s in [0.5, 1.0, 2.0, 4.0]
    h_true = true_h23(s)
    h_fit = h23_haz_pen(s, pars_pen.h23, NamedTuple())
    ratio = h_fit / h_true
    println("  s=$s: true=$(round(h_true, digits=3)), fitted=$(round(h_fit, digits=3)), ratio=$(round(ratio, digits=2))")
end

pars_pen = MultistateModels.get_parameters(fitted_pen)
h23_haz_pen = fitted_pen.hazards[fitted_pen.hazkeys[:h23]]

println("--- Penalized (λ=0.01) h23 Comparison ---")
for s in [0.5, 1.0, 2.0, 4.0]
    h_true = true_h23(s)
    h_fit = h23_haz_pen(s, pars_pen.h23, NamedTuple())
    ratio = h_fit / h_true
    println("  s=$s: true=$(round(h_true, digits=3)), fitted=$(round(h_fit, digits=3)), ratio=$(round(ratio, digits=2))")
end

# Compare h12 and h13
println("\n--- Summary Table ---")
println("\nUnpenalized fit:")
h12_unpen = fitted_unpen.hazards[fitted_unpen.hazkeys[:h12]]
h13_unpen = fitted_unpen.hazards[fitted_unpen.hazkeys[:h13]]
for t in [0.5, 2.0, 4.0]
    r12 = h12_unpen(t, pars_unpen.h12, NamedTuple()) / true_h12(t)
    r13 = h13_unpen(t, pars_unpen.h13, NamedTuple()) / true_h13(t)
    r23 = h23_haz_unpen(t, pars_unpen.h23, NamedTuple()) / true_h23(t)
    println("  t/s=$t: h12=$(round(r12,digits=2)), h13=$(round(r13,digits=2)), h23=$(round(r23,digits=2))")
end
