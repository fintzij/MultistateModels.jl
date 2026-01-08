# =============================================================================
# Hazard Generator Functions
# =============================================================================
#
# Runtime code generation for parametric hazard functions. These generators
# create specialized hazard_fn and cumhaz_fn for each distribution family.
#
# =============================================================================

"""
    generate_exponential_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for exponential hazards with optional PH/AFT covariate
effects controlled by `linpred_effect`.

PARAMETER CONVENTION: Receives natural-scale baseline parameters (exp already applied).
- `pars.baseline.xxx` is the natural-scale rate (positive)
- Covariate coefficients are on natural scale (unconstrained)
- Formula: h(t|x) = rate * exp(β'x) for PH, h(t|x) = rate * exp(-β'x) for AFT
"""
function generate_exponential_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr_named(parnames)
    
    # Extract baseline parameter name (should be :h*_Rate)
    baseline_parname = parnames[1]
    
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                linear_pred = $linear_pred_expr
                # pars.baseline is on NATURAL scale (no exp needed)
                return pars.baseline.$(baseline_parname) * exp(linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                linear_pred = $linear_pred_expr
                return pars.baseline.$(baseline_parname) * exp(linear_pred) * (ub - lb)
            end
        ))
    elseif linpred_effect == :aft
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                linear_pred = $linear_pred_expr
                return pars.baseline.$(baseline_parname) * exp(-linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                linear_pred = $linear_pred_expr
                return pars.baseline.$(baseline_parname) * exp(-linear_pred) * (ub - lb)
            end
        ))
    else
        throw(ArgumentError("Unsupported linpred_effect :$(linpred_effect) for exponential hazard. Supported: :ph, :aft"))
    end

    return hazard_fn, cumhaz_fn
end

"""
    generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for Weibull hazards, supporting PH or AFT covariate effects.

PARAMETER CONVENTION: Receives natural-scale baseline parameters.
- `pars.baseline.shape` is the natural-scale shape parameter (positive)
- `pars.baseline.scale` is the natural-scale scale parameter (positive)
- Formula: h(t) = shape * scale * t^(shape-1) for baseline Weibull
"""
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr_named(parnames)
    
    # Extract baseline parameter names (should be :h*_shape and :h*_scale)
    shape_parname = parnames[1]
    scale_parname = parnames[2]
    
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                # pars.baseline is on NATURAL scale (no exp needed)
                shape = pars.baseline.$(shape_parname)
                scale = pars.baseline.$(scale_parname)
                linear_pred = $linear_pred_expr
                
                # h(t) = shape * scale * t^(shape-1) * exp(linear_pred)
                haz = shape * scale * exp(linear_pred)
                if shape != 1.0
                    haz *= t^(shape - 1)
                end
                
                return haz
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                shape = pars.baseline.$(shape_parname)
                scale = pars.baseline.$(scale_parname)
                linear_pred = $linear_pred_expr
                return scale * exp(linear_pred) * (ub^shape - lb^shape)
            end
        ))
    elseif linpred_effect == :aft
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                shape = pars.baseline.$(shape_parname)
                scale = pars.baseline.$(scale_parname)
                linear_pred = $linear_pred_expr
                
                # AFT: h(t|x) = h_0(t * exp(-linear_pred)) * exp(-linear_pred)
                # = shape * scale * t^(shape-1) * exp(-shape * linear_pred)
                haz = shape * scale * exp(-shape * linear_pred)
                if shape != 1.0
                    haz *= t^(shape - 1)
                end
                
                return haz
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                shape = pars.baseline.$(shape_parname)
                scale = pars.baseline.$(scale_parname)
                linear_pred = $linear_pred_expr
                return scale * exp(-shape * linear_pred) * (ub^shape - lb^shape)
            end
        ))
    else
        throw(ArgumentError("Unsupported linpred_effect :$(linpred_effect) for Weibull hazard. Supported: :ph, :aft"))
    end

    return hazard_fn, cumhaz_fn
end

"""
    generate_gompertz_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for Gompertz hazards with PH/AFT covariate handling.

# Parameterization

Matches the **flexsurv** R package parameterization:

- **Hazard**: h(t) = rate × exp(shape × t)
- **Cumulative hazard**: H(t) = (rate/shape) × (exp(shape×t) - 1) for shape ≠ 0
- **Survival**: S(t) = exp(-H(t))

# Parameters

- `shape` (unconstrained): Controls how hazard changes over time
  - shape > 0: hazard increases exponentially (typical aging/wear-out)
  - shape = 0: constant hazard (reduces to exponential with rate parameter)
  - shape < 0: hazard decreases over time (defective/cure models)
- `rate` (positive): Baseline hazard rate at t=0

# Storage Convention

- **Estimation scale**: `[shape, log(rate)]` — shape is unconstrained, rate is log-transformed
- **Natural scale**: `[shape, rate]` — shape unchanged, rate exponentiated

# Covariate Effects

- `:ph` (proportional hazards): h(t|x) = rate × exp(shape×t + β'x)
- `:aft` (accelerated failure time): h(t|x) = rate × exp(shape×t×exp(-β'x)) × exp(-β'x)

# Default Initialization

When created via `multistatemodel()`, Gompertz hazards are initialized with:
- shape = 0 (so hazard starts as constant/exponential)
- rate = crude transition rate from data

This ensures sensible starting values for optimization.

# Reference

Jackson, C. (2016). flexsurv: A Platform for Parametric Survival Modeling in R.
Journal of Statistical Software, 70(8), 1-33.
"""
function generate_gompertz_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr_named(parnames)
    
    # Extract baseline parameter names (should be :h*_shape and :h*_rate)
    shape_parname = parnames[1]
    rate_parname = parnames[2]
    
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                # pars.baseline is on NATURAL scale
                # flexsurv parameterization: h(t) = rate * exp(shape * t)
                shape = pars.baseline.$(shape_parname)
                rate = pars.baseline.$(rate_parname)
                linear_pred = $linear_pred_expr
                return rate * exp(shape * t + linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                # flexsurv parameterization: H(t) = (rate/shape) * (exp(shape*t) - 1)
                shape = pars.baseline.$(shape_parname)
                rate = pars.baseline.$(rate_parname)
                if abs(shape) < $SHAPE_ZERO_TOL
                    # Reduces to exponential: H = rate * (ub - lb)
                    baseline_cumhaz = rate * (ub - lb)
                else
                    baseline_cumhaz = (rate / shape) * (exp(shape * ub) - exp(shape * lb))
                end
                linear_pred = $linear_pred_expr
                return baseline_cumhaz * exp(linear_pred)
            end
        ))
    elseif linpred_effect == :aft
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                # flexsurv parameterization: h(t) = rate * exp(shape * t)
                shape = pars.baseline.$(shape_parname)
                rate = pars.baseline.$(rate_parname)
                linear_pred = $linear_pred_expr
                time_scale = exp(-linear_pred)
                # AFT: h(t|x) = h_0(t * exp(-linear_pred)) * exp(-linear_pred)
                return rate * exp(shape * t * time_scale) * time_scale
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                # flexsurv parameterization with AFT time scaling
                shape = pars.baseline.$(shape_parname)
                rate = pars.baseline.$(rate_parname)
                linear_pred = $linear_pred_expr
                time_scale = exp(-linear_pred)
                scaled_shape = shape * time_scale
                scaled_rate = rate * time_scale
                if abs(scaled_shape) < $SHAPE_ZERO_TOL
                    baseline_cumhaz = scaled_rate * (ub - lb)
                else
                    baseline_cumhaz = (scaled_rate / scaled_shape) * (exp(scaled_shape * ub) - exp(scaled_shape * lb))
                end
                return baseline_cumhaz
            end
        ))
    else
        throw(ArgumentError("Unsupported linpred_effect :$(linpred_effect) for Gompertz hazard. Supported: :ph, :aft"))
    end

    return hazard_fn, cumhaz_fn
end

"""
    generate_phasetype_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for phase-type (Coxian) hazards used in expanded Markov models.

Phase-type hazards are exponential hazards used in the expanded state space.
They represent transitions between phases within a state, or exit from the state.

# Note
This is called during phase-type expansion to create individual phase hazards.
The full phase-type distribution behavior emerges from the Markov model structure.
"""
function generate_phasetype_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    # Phase-type hazards are just exponential hazards in the expanded space
    return generate_exponential_hazard(parnames, linpred_effect)
end
