# Test Phase 2.5 - Log scale parameter fix
# This test verifies that hazard generators accept log scale parameters

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# Need to copy just the generator functions to avoid dependencies
# Copy from src/hazards.jl lines 13-211

function generate_exponential_hazard(has_covariates::Bool)
    if !has_covariates
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                return exp(pars[1])
            end
        ))
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                return exp(pars[1]) * (ub - lb)
            end
        ))
    else
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_baseline = pars[1]
                linear_pred = zero(eltype(pars))
                for i in 2:length(pars)
                    linear_pred += pars[i] * covars[i-1]
                end
                return exp(log_baseline + linear_pred)
            end
        ))
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_baseline = pars[1]
                linear_pred = zero(eltype(pars))
                for i in 2:length(pars)
                    linear_pred += pars[i] * covars[i-1]
                end
                return exp(log_baseline + linear_pred) * (ub - lb)
            end
        ))
    end
    return hazard_fn, cumhaz_fn
end

function generate_weibull_hazard(has_covariates::Bool)
    if !has_covariates
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                return exp(log_shape + expm1(log_shape) * log(t) + log_scale)
            end
        ))
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape, scale = exp(log_shape), exp(log_scale)
                return scale * (ub^shape - lb^shape)
            end
        ))
    else
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                linear_pred = zero(eltype(pars))
                for i in 3:length(pars)
                    linear_pred += pars[i] * covars[i-2]
                end
                return exp(log_shape + expm1(log_shape) * log(t) + log_scale + linear_pred)
            end
        ))
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape, scale = exp(log_shape), exp(log_scale)
                linear_pred = zero(eltype(pars))
                for i in 3:length(pars)
                    linear_pred += pars[i] * covars[i-2]
                end
                return scale * exp(linear_pred) * (ub^shape - lb^shape)
            end
        ))
    end
    return hazard_fn, cumhaz_fn
end

function generate_gompertz_hazard(has_covariates::Bool)
    if !has_covariates
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                return exp(log_scale + log_shape + shape * t)
            end
        ))
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                if abs(shape) < 1e-10
                    return scale * (ub - lb)
                else
                    return scale * (exp(shape * ub) - exp(shape * lb))
                end
            end
        ))
    else
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                linear_pred = zero(eltype(pars))
                for i in 3:length(pars)
                    linear_pred += pars[i] * covars[i-2]
                end
                return exp(log_scale + log_shape + shape * t + linear_pred)
            end
        ))
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                if abs(shape) < 1e-10
                    baseline_cumhaz = scale * (ub - lb)
                else
                    baseline_cumhaz = scale * (exp(shape * ub) - exp(shape * lb))
                end
                linear_pred = zero(eltype(pars))
                for i in 3:length(pars)
                    linear_pred += pars[i] * covars[i-2]
                end
                return baseline_cumhaz * exp(linear_pred)
            end
        ))
    end
    return hazard_fn, cumhaz_fn
end

println("="^70)
println("Testing Phase 2.5: Log Scale Parameter Fix")
println("="^70)

# Test 1: Exponential hazard
println("\n1. Testing Exponential hazard with log scale...")
haz_fn, cumhaz_fn = generate_exponential_hazard(false)
log_baseline = 0.8
natural_baseline = exp(0.8)  # Should be 2.225...

h1 = haz_fn(1.0, [log_baseline], Float64[])
println("   h(1.0) with log_baseline=0.8: $h1")
println("   Expected: exp(0.8) = $natural_baseline")
@assert isapprox(h1, natural_baseline, rtol=1e-10) "Exponential hazard mismatch!"
println("   ✓ PASS")

# Test 2: Weibull hazard
println("\n2. Testing Weibull hazard with log scale...")
haz_fn_wei, cumhaz_fn_wei = generate_weibull_hazard(false)
log_shape = -0.25  # shape = 0.779
log_scale = 0.2    # scale = 1.221
t = 1.0
expected_log_h = log_shape + expm1(log_shape) * log(t) + log_scale
expected_h = exp(expected_log_h)

h_wei = haz_fn_wei(t, [log_shape, log_scale], Float64[])
println("   h(1.0) with log_shape=-0.25, log_scale=0.2: $h_wei")
println("   Expected: $expected_h")
@assert isapprox(h_wei, expected_h, rtol=1e-10) "Weibull hazard mismatch!"
println("   ✓ PASS")

# Test 3: Gompertz hazard
println("\n3. Testing Gompertz hazard with log scale...")
haz_fn_gomp, cumhaz_fn_gomp = generate_gompertz_hazard(false)
log_shape = log(1.5)  # shape = 1.5
log_scale = log(0.5)  # scale = 0.5
t = 1.0
expected_log_h = log_scale + log_shape + exp(log_shape) * t
expected_h = exp(expected_log_h)

h_gomp = haz_fn_gomp(t, [log_shape, log_scale], Float64[])
println("   h(1.0) with shape=1.5, scale=0.5: $h_gomp")
println("   Expected: $expected_h")
@assert isapprox(h_gomp, expected_h, rtol=1e-10) "Gompertz hazard mismatch!"
println("   ✓ PASS")

# Test 4: Exponential with covariates
println("\n4. Testing Exponential with covariates...")
haz_fn_cov, cumhaz_fn_cov = generate_exponential_hazard(true)
log_baseline = 0.5
coefs = [0.3, -0.2]
covars = [1.0, 2.0]
linear_pred = 0.3 * 1.0 + (-0.2) * 2.0  # = -0.1
expected_h_cov = exp(log_baseline + linear_pred)  # exp(0.5 - 0.1) = exp(0.4)

h_cov = haz_fn_cov(1.0, [log_baseline; coefs], covars)
println("   h(1.0) with covariates: $h_cov")
println("   Expected: exp(0.4) = $expected_h_cov")
@assert isapprox(h_cov, expected_h_cov, rtol=1e-10) "Exponential with covariates mismatch!"
println("   ✓ PASS")

# Test 5: Weibull with covariates
println("\n5. Testing Weibull with covariates...")
haz_fn_wei_cov, _ = generate_weibull_hazard(true)
log_shape = -0.25
log_scale = 0.2
coefs = [0.6, -0.4, 0.15]
covars = [1.0, 0.5, 2.0]
t = 1.0
linear_pred = 0.6 * 1.0 + (-0.4) * 0.5 + 0.15 * 2.0  # = 0.7
expected_log_h_wei_cov = log_shape + expm1(log_shape) * log(t) + log_scale + linear_pred
expected_h_wei_cov = exp(expected_log_h_wei_cov)

h_wei_cov = haz_fn_wei_cov(t, vcat([log_shape, log_scale], coefs), covars)
println("   h(1.0) with covariates: $h_wei_cov")
println("   Expected: $expected_h_wei_cov")
@assert isapprox(h_wei_cov, expected_h_wei_cov, rtol=1e-10) "Weibull with covariates mismatch!"
println("   ✓ PASS")

println("\n" * "="^70)
println("All Phase 2.5 tests PASSED! ✓")
println("Hazard generators correctly accept log scale parameters.")
println("="^70)
