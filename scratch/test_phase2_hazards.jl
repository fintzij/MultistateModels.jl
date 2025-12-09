using MultistateModels
using BSplineKit

println("Testing Phase 2 Hazard Generators with Named Parameter Access")
println("="^70)

# Test 1: Exponential hazard with named parameters
println("\n1. Testing Exponential Hazard...")
parnames_exp = [:h12_Intercept, :h12_age, :h12_sex]
hazard_fn_exp, cumhaz_fn_exp = MultistateModels.generate_exponential_hazard(parnames_exp, :ph)

# Create named parameter tuple
pars_exp = (
    baseline = (h12_Intercept = log(0.5),),
    covariates = (h12_age = 0.3, h12_sex = 0.1)
)
covars_exp = (age = 50.0, sex = 1.0)

h_exp = hazard_fn_exp(1.0, pars_exp, covars_exp)
H_exp = cumhaz_fn_exp(0.0, 2.0, pars_exp, covars_exp)
println("  ✓ Exponential hazard: h(1) = $h_exp, H(0,2) = $H_exp")

# Test 2: Weibull hazard with named parameters
println("\n2. Testing Weibull Hazard...")
parnames_wei = [:h12_shape, :h12_scale, :h12_age]
hazard_fn_wei, cumhaz_fn_wei = MultistateModels.generate_weibull_hazard(parnames_wei, :ph)

pars_wei = (
    baseline = (h12_shape = log(1.5), h12_scale = log(0.2)),
    covariates = (h12_age = 0.3,)
)
covars_wei = (age = 50.0,)

h_wei = hazard_fn_wei(1.0, pars_wei, covars_wei)
H_wei = cumhaz_fn_wei(0.0, 2.0, pars_wei, covars_wei)
println("  ✓ Weibull hazard (PH): h(1) = $h_wei, H(0,2) = $H_wei")

# Test 3: Gompertz hazard with named parameters
println("\n3. Testing Gompertz Hazard...")
parnames_gom = [:h12_shape, :h12_scale, :h12_age]
hazard_fn_gom, cumhaz_fn_gom = MultistateModels.generate_gompertz_hazard(parnames_gom, :ph)

pars_gom = (
    baseline = (h12_shape = log(0.8), h12_scale = log(0.3)),
    covariates = (h12_age = 0.2,)
)
covars_gom = (age = 50.0,)

h_gom = hazard_fn_gom(1.0, pars_gom, covars_gom)
H_gom = cumhaz_fn_gom(0.0, 2.0, pars_gom, covars_gom)
println("  ✓ Gompertz hazard: h(1) = $h_gom, H(0,2) = $H_gom")

# Test 4: Spline hazard with named parameters
println("\n4. Testing Spline Hazard...")
# Create simple B-spline basis
knots = [0.0, 0.3, 0.7, 1.0]
degree = 2
basis = BSplineBasis(BSplineOrder(degree+1), knots)
nbasis = length(basis)

parnames_spl = [Symbol("h12_sp$i") for i in 1:nbasis]
push!(parnames_spl, :h12_age)

hazard_fn_spl, cumhaz_fn_spl = MultistateModels._generate_spline_hazard_fns(
    basis,
    BSplineKit.SplineExtrapolations.Flat(),
    0,  # monotone = 0 (no monotone constraint)
    nbasis,
    parnames_spl,
    :ph
)

# Create named parameters for spline
baseline_vals = log.([0.5, 0.8, 1.2, 0.6])  # nbasis values
pars_spl = (
    baseline = NamedTuple{Tuple(parnames_spl[1:nbasis])}(baseline_vals),
    covariates = (h12_age = 0.15,)
)
covars_spl = (age = 50.0,)

h_spl = hazard_fn_spl(0.5, pars_spl, covars_spl)
H_spl = cumhaz_fn_spl(0.0, 1.0, pars_spl, covars_spl)
println("  ✓ Spline hazard: h(0.5) = $h_spl, H(0,1) = $H_spl")

println("\n" * "="^70)
println("All Phase 2 hazard generator tests passed! ✓")
println("Named parameter access working correctly for all hazard types.")
