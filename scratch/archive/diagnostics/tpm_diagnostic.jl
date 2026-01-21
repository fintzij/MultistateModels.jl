using MultistateModels
using DataFrames
import MultistateModels: Hazard, multistatemodel, set_parameters!, get_parameters, cumulative_hazard

# Minimal setup
MY_DEGREE = 2
MY_BKNOTS = [0.0, 5.0]
MY_COEFS_H12 = [0.15, 0.25]
MY_COEFS_H23 = [0.10, 0.20]
knots = [2.5]

template = DataFrame(
    id = [1, 1],
    tstart = [0.0, 2.0],
    tstop = [2.0, 4.0],
    statefrom = [1, 2],
    stateto = [2, 3],
    obstype = [1, 1]
)

h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
    degree=MY_DEGREE, knots=knots, boundaryknots=MY_BKNOTS, extrapolation="constant")
h23_sp = Hazard(@formula(0 ~ 1), "sp", 2, 3; 
    degree=MY_DEGREE, knots=knots, boundaryknots=MY_BKNOTS, extrapolation="constant")

model = multistatemodel(h12_sp, h23_sp; data=template)
set_parameters!(model, (h12 = MY_COEFS_H12, h23 = MY_COEFS_H23))

pars_h12 = collect(MY_COEFS_H12)
pars_h23 = collect(MY_COEFS_H23)

# Compute cumulative hazard at various time points
println("=== Cumulative hazard H12(0, t) ===")
for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0]
    H = cumulative_hazard(model.hazards[1], 0.0, t, NamedTuple())
    println("  t=$t: H12(0,t) = ", round(H, digits=4))
end

println("\n=== Cumulative hazard H23(0, t) ===")
for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0]
    H = cumulative_hazard(model.hazards[2], 0.0, t, NamedTuple())
    println("  t=$t: H23(0,t) = ", round(H, digits=4))
end

# Check hazard at late times
println("\n=== Hazard h12(t) ===")
for t in [0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
    h = model.hazards[1](t, pars_h12, NamedTuple())
    println("  t=$t: h12(t) = ", round(h, digits=4))
end

println("\n=== Hazard h23(t) ===")
for t in [0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
    h = model.hazards[2](t, pars_h23, NamedTuple())
    println("  t=$t: h23(t) = ", round(h, digits=4))
end

println("\n=== Checking extended knots ===")
# Access the hazard's internal basis info
println("h12 knots: ", model.hazards[1].knots)
println("h23 knots: ", model.hazards[2].knots)
