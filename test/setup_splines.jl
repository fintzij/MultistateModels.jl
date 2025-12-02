# set up a MultistateModel object with spline hazards
using BSplineKit
using DataFrames
using Distributions
using MultistateModels
using Random

# Create test data with covariates
dat = DataFrame(
    id = [1, 1, 2, 2],
    tstart = [0.0, 0.5, 0.0, 0.3],
    tstop = [0.5, 1.0, 0.3, 1.0],
    statefrom = [1, 2, 1, 3],
    stateto = [2, 1, 3, 1],
    obstype = [1, 1, 1, 1],
    x = [0.5, 0.5, -0.3, -0.3]
)

# Create multistate model object with various spline configurations
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
             degree=3, knots=collect(0.2:0.2:0.8), 
             extrapolation="flat", natural_spline=true)

h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; 
             degree=1, knots=[0.25, 0.5, 0.75])

h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
             degree=3, knots=collect(0.2:0.2:0.8))

h31 = Hazard(@formula(0 ~ 1 + x), "sp", 3, 1; 
             degree=1, knots=[0.25, 0.5, 0.75], 
             extrapolation="flat")

h32 = Hazard(@formula(0 ~ 1), "sp", 3, 2; 
             degree=1, extrapolation="linear")

hazards = (h12, h13, h21, h31, h32)
splinemod = multistatemodel(h12, h13, h21, h31, h32; data=dat)

# Initialize with random parameters
Random.seed!(12345)
for (h, haz) in enumerate(splinemod.hazards)
    npar = haz.npar_total
    new_pars = rand(Normal(0, 0.5), npar)
    set_parameters!(splinemod, h, new_pars)
end

# Test that hazard evaluation works
function test_spline_evaluation(model)
    for (i, haz) in enumerate(model.hazards)
        # Get log-scale parameters (what hazard functions expect)
        pars = get_parameters(model, i, scale=:log)
        
        # Test hazard at several times
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]
            h_val = haz(t, pars, NamedTuple())
            @assert isfinite(h_val) "Hazard $i at t=$t is not finite: $h_val"
            @assert h_val >= 0 "Hazard $i at t=$t is negative: $h_val"
        end
        
        
        # Test cumulative hazard
        H_val = MultistateModels.cumulative_hazard(haz, 0.0, 1.0, pars, NamedTuple())
        @assert isfinite(H_val) "Cumulative hazard $i is not finite: $H_val"
        @assert H_val >= 0 "Cumulative hazard $i is negative: $H_val"
    end
    return true
end

# Run the test
@assert test_spline_evaluation(splinemod) "Spline evaluation test failed"

# ================================================================================
# Test automatic knot placement
# ================================================================================

# Create larger dataset for auto knot placement
# Need properly sorted intervals per subject
Random.seed!(42)
n_subjects = 50

function generate_auto_test_data(n_subjects)
    rows = []
    for subj in 1:n_subjects
        # First interval: state 1 -> 2
        t1 = rand(Uniform(0.3, 0.8))
        push!(rows, (id=subj, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1, x=randn()))
        # Second interval: state 2 -> 1
        t2 = t1 + rand(Uniform(0.2, 0.6))
        push!(rows, (id=subj, tstart=t1, tstop=t2, statefrom=2, stateto=1, obstype=1, x=rows[end].x))
    end
    return DataFrame(rows)
end

auto_dat = generate_auto_test_data(n_subjects)

# Create model with automatic knot placement (knots=nothing)
h12_auto = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                   degree=3, knots=nothing, natural_spline=true)
h21_auto = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
                   degree=3, knots=nothing)

auto_model = multistatemodel(h12_auto, h21_auto; data=auto_dat)

# Test that automatic knots were placed
for (i, haz) in enumerate(auto_model.hazards)
    @assert length(haz.knots) > 2 "Expected auto knots but got only boundary knots for hazard $i"
    # Knots should be sorted
    @assert issorted(haz.knots) "Knots not sorted for hazard $i"
    # Interior knots should be within boundaries
    interior = haz.knots[2:end-1]
    @assert all(interior .> haz.knots[1]) "Interior knots below lower boundary for hazard $i"
    @assert all(interior .< haz.knots[end]) "Interior knots above upper boundary for hazard $i"
end

# Test evaluation with auto-placed knots
@assert test_spline_evaluation(auto_model) "Auto knot model evaluation failed"

# ================================================================================
# Test time transformation support for splines
# ================================================================================

# Create a spline hazard with time_transform=true
h12_tt = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                 degree=3, knots=[0.3, 0.5, 0.7],
                 natural_spline=true,
                 time_transform=true)
h21_tt = Hazard(@formula(0 ~ 1), "sp", 2, 1; 
                 degree=3, knots=[0.3, 0.5, 0.7],
                 time_transform=true)

tt_model = multistatemodel(h12_tt, h21_tt; data=auto_dat)

# Test that time transform evaluation works
function test_time_transform_spline(model)
    for (i, haz) in enumerate(model.hazards)
        pars = get_parameters(model, i, scale=:log)
        
        # Create covariates NamedTuple
        covars = if haz.has_covariates
            (x = 0.5,)
        else
            NamedTuple()
        end
        
        # Compute linear predictor
        linpred = MultistateModels._linear_predictor(pars, covars, haz)
        
        # Test _time_transform_hazard
        for t in [0.1, 0.5, 0.9]
            h_val = MultistateModels._time_transform_hazard(haz, pars, t, linpred)
            @assert isfinite(h_val) "Time transform hazard $i at t=$t is not finite: $h_val"
            @assert h_val >= 0 "Time transform hazard $i at t=$t is negative: $h_val"
        end
        
        # Test _time_transform_cumhaz
        H_val = MultistateModels._time_transform_cumhaz(haz, pars, 0.0, 1.0, linpred)
        @assert isfinite(H_val) "Time transform cumhaz $i is not finite: $H_val"
        @assert H_val >= 0 "Time transform cumhaz $i is negative: $H_val"
    end
    return true
end

@assert test_time_transform_spline(tt_model) "Time transform spline test failed"

# Test that the _maybe_transform functions work through the full call path
function test_full_transform_path(model)
    for (i, haz) in enumerate(model.hazards)
        pars = get_parameters(model, i, scale=:log)
        covars = haz.has_covariates ? (x = 0.3,) : NamedTuple()
        
        # Call through the transform path
        h_val = MultistateModels._maybe_transform_hazard(
            haz, pars, covars, 0.5;
            apply_transform=true,
            cache_context=nothing,
            hazard_slot=i
        )
        @assert isfinite(h_val) "Full path hazard $i is not finite"
        
        H_val = MultistateModels._maybe_transform_cumulhaz(
            haz, pars, covars, 0.0, 1.0;
            apply_transform=true,
            cache_context=nothing,
            hazard_slot=i
        )
        @assert isfinite(H_val) "Full path cumhaz $i is not finite"
    end
    return true
end

@assert test_full_transform_path(tt_model) "Full transform path test failed"

println("Setup splines test passed!")
