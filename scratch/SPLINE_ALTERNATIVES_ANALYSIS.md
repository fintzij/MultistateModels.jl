# Analysis: Spline Backend Alternatives

## Current Implementation (BSplineKit.jl)

### What We Use
- **B-spline basis construction** with custom knot placement
- **Natural spline recombination** via `RecombinedBSplineBasis`
- **Extrapolation methods**: Linear and Flat
- **Integral computation**: `integral()` for cumulative hazards
- **Derivative computation**: `diff()` for risk period calculations
- **Order specification**: Up to cubic (degree 3)
- **Monotonicity constraints**: Custom transformation system

### Current Dependencies
```julia
BSplineKit = "093aae92-e908-43d7-9660-e50ee39d5a0a"
```

---

## Option 1: DataInterpolations.jl

### Migration Feasibility: **MODERATE** ⚠️

### Advantages ✅
1. **SciML ecosystem integration**
   - Better integration with DifferentialEquations.jl
   - Symbolics.jl compatibility for automatic differentiation
   - Part of the broader SciML stack
   
2. **Simpler API**
   ```julia
   # BSplineKit (current)
   B = BSplineBasis(BSplineOrder(degree + 1), knots)
   B_recomb = RecombinedBSplineBasis(B, Natural())
   sphaz = SplineExtrapolation(Spline(undef, B), extrap_method)
   
   # DataInterpolations (proposed)
   interp = CubicSpline(coefs, knots)  # Much simpler!
   ```

3. **Built-in regularization**
   - `RegularizationSmooth` for penalized likelihood approaches
   - Could replace manual monotonicity constraints
   
4. **Performance**
   - Generally faster for evaluation
   - Optimized for scientific computing workflows

### Disadvantages ❌
1. **Missing features we currently use**
   - ❌ **No natural spline recombination** (our custom `RecombinedBSplineBasis`)
   - ❌ **No explicit monotonicity constraints** (we implement custom transformations)
   - ❌ **Limited B-spline customization** (degree, knot vector control)
   - ⚠️ **Extrapolation**: Has it but API differs significantly
   
2. **Would require custom implementation**
   - Natural spline constraints (boundary condition that 2nd derivative = 0)
   - Monotonicity transformations (`spline_ests2coefs`, `spline_coefs2ests`)
   - Risk period calculation logic
   
3. **Loss of fine-grained control**
   - BSplineKit gives us direct access to basis matrices
   - Our `RecombinedBSplineBasis` approach for natural splines is sophisticated
   - Custom knot placement strategies well-supported in BSplineKit

### Migration Effort: **MEDIUM-HIGH**
- Need to reimplement natural spline logic
- Need to reimplement monotonicity constraints
- Need to verify integral/derivative calculations
- **Estimated**: 3-5 days of work + extensive testing

---

## Option 2: Sparse Gaussian Processes

### Feasibility: **HIGH** 🎯 (Excellent fit for survival analysis!)

### Why GPs Are Attractive for Hazards

1. **Theoretical advantages**
   - Naturally enforce smoothness via covariance functions
   - Built-in uncertainty quantification
   - Can encode monotonicity through the choice of kernel
   - Flexible prior on baseline hazard shape
   
2. **Sparse GP methods are mature**
   - Variational Sparse GPs (Titsias, 2009)
   - FITC (Fully Independent Training Conditional)
   - VFE (Variational Free Energy)
   - Inducing points reduce O(n³) to O(nm²) where m << n

### Implementation Options

#### Option 2A: AbstractGPs.jl (Recommended)
```julia
using AbstractGPs
using KernelFunctions

# Define GP prior on log-hazard
kernel = Matern52Kernel() ∘ ScaleTransform(ℓ)  # length scale ℓ
f = GP(kernel)

# Sparse approximation with inducing points
m = 20  # Number of inducing points
z = range(minimum(t), maximum(t), length=m)  # Inducing locations
sparse_f = SparseVariationalApproximation(f, z)

# Inference (variational or exact)
fx = sparse_f(X, noise_var)
```

**Advantages:**
- ✅ Well-maintained, active development
- ✅ Composable kernels (sum/product/transforms)
- ✅ GPU support via KernelFunctions.jl
- ✅ Automatic differentiation friendly
- ✅ Sparse GP implementations available
- ✅ Integrates with Stheno.jl for multi-output GPs (multi-state!)

**Disadvantages:**
- ⚠️ Need to implement cumulative hazard (integral of exp(GP))
- ⚠️ Requires numerical integration (QuadGK.jl - already a dependency!)
- ⚠️ Different parameterization than current approach

#### Option 2B: GaussianProcesses.jl
```julia
using GaussianProcesses

# Define mean and kernel
m = MeanZero()
k = SE(0.0, 0.0)  # Squared exponential

# Fit sparse GP
gp = GP(X, y, m, k, logObsNoise)
optimize!(gp)  # MLE for hyperparameters
```

**Advantages:**
- ✅ Mature, stable package
- ✅ Built-in optimization for hyperparameters
- ✅ Various kernel implementations

**Disadvantages:**
- ⚠️ Less actively maintained than AbstractGPs.jl
- ⚠️ Smaller ecosystem
- ⚠️ Less flexible for custom likelihood functions

### Mathematical Formulation

For transition i→j, model log-baseline hazard as:
```
log h₀(t) ~ GP(μ(t), k(t, t'))
```

**Kernel choices:**
1. **Matérn (ν=5/2)**: Twice differentiable, good default
2. **Matérn (ν=3/2)**: Once differentiable, smoother
3. **Squared Exponential**: Infinitely differentiable, very smooth
4. **Periodic**: For cyclic patterns (e.g., seasonal effects)

**Monotonicity enforcement:**
- Use derivative observations: GP on h'(t) constrained to be positive
- OR: GP on log(cumulative hazard) (automatically monotone)
- OR: Transformation approach similar to current splines

**Sparse GP for computational efficiency:**
```
Given data at n timepoints
Choose m << n inducing points
Complexity: O(nm²) instead of O(n³)
For n=1000, m=50: ~400× speedup
```

### Integration with Current Code

**What changes:**
```julia
# Current (splines)
struct SplineHazard
    hazsp::SplineExtrapolation     # B-spline evaluator
    chazsp::Spline                  # Cumulative hazard spline
    # ... other fields
end

# Proposed (GP)
struct GPHazard  
    gp::AbstractGPs.PosteriorGP     # Sparse GP on log-hazard
    cumhaz_integrator::Function     # Numerical integrator for H(t)
    inducing_points::Vector{Float64}
    kernel::Kernel
    # ... other fields
end
```

**Hazard evaluation:**
```julia
# Current
h(t) = hazsp(t)                    # Direct B-spline evaluation

# Proposed  
h(t) = exp(mean(gp(t)))            # Exponentiate GP mean
```

**Cumulative hazard:**
```julia
# Current
H(lb, ub) = chazsp(ub) - chazsp(lb)  # Analytic spline integral

# Proposed
H(lb, ub) = quadgk(t -> exp(mean(gp(t))), lb, ub)[1]  # Numerical integration
```

### Performance Considerations

**Pros:**
- Sparse GPs scale well: O(nm²) for m inducing points
- Can cache GP posterior for repeated evaluations
- Uncertainty quantification "for free"

**Cons:**
- Numerical integration slower than analytical spline integrals
- Need to tune number of inducing points (m)
- Hyperparameter optimization adds overhead

**Mitigation strategies:**
1. Cache cumulative hazard evaluations on grid
2. Use adaptive quadrature only when needed
3. Pre-compute GP posterior once per optimization step
4. Consider Gauss-Hermite quadrature for speed

---

## Option 3: Hybrid Approach 🎯 (RECOMMENDED)

### Keep BSplineKit for now, add GP option

```julia
@enum BaselineType begin
    SplineBaseline
    GPBaseline  
end

struct ParametricHazard
    baseline_type::BaselineType
    # ... existing fields for splines
    # ... new fields for GP
end
```

**Advantages:**
- ✅ No breaking changes to existing code
- ✅ Users can choose splines (fast, proven) or GPs (flexible, uncertainty)
- ✅ Can benchmark both approaches
- ✅ Future-proof architecture

**Implementation plan:**
1. Abstract out baseline hazard interface
2. Implement GP backend alongside spline backend
3. Common API for h(t), H(a,b), parameter updates
4. Let users experiment with both

---

## Comparison Table

| Feature | BSplineKit | DataInterpolations | Sparse GPs |
|---------|-----------|-------------------|------------|
| **Current fit** | ✅ Perfect | ⚠️ Good | ⚠️ Requires work |
| **Natural splines** | ✅ Native | ❌ Manual | N/A |
| **Monotonicity** | ✅ Custom | ❌ Manual | ⚠️ Tricky |
| **Analytic integrals** | ✅ Yes | ✅ Yes | ❌ Numerical |
| **Uncertainty quantification** | ❌ No | ❌ No | ✅ Built-in |
| **Flexibility** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **SciML ecosystem** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Migration effort** | 0 days | 3-5 days | 5-10 days |
| **Novel contribution** | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

---

## Recommendations

### Short-term (Current Branch)
**KEEP BSplineKit.jl**
- ✅ It works perfectly for current needs
- ✅ Natural spline implementation is sophisticated
- ✅ Monotonicity constraints working well
- ✅ Fast analytical integrals critical for performance

### Medium-term (6-12 months)
**Implement Sparse GPs as experimental feature**
- Start with AbstractGPs.jl
- Implement for 1-2 simple transitions
- Compare with splines on benchmark datasets
- Could be a strong methodological contribution!

### Long-term Vision
**Unified baseline hazard interface with multiple backends:**
```julia
# User chooses at model specification
hazard(1, 2, Weibull())              # Parametric
hazard(1, 2, Spline(degree=3))       # Current approach
hazard(1, 2, GP(kernel=Matern52()))  # New GP approach
```

---

## Why Sparse GPs Are Exciting for This Application

1. **Methodological novelty**: Few survival analysis packages offer GP baselines
2. **Uncertainty quantification**: Automatically get credible intervals on hazards
3. **Multi-state synergy**: GPs naturally extend to multi-output problems (Stheno.jl)
4. **Prior knowledge**: Can encode domain knowledge through kernel choice
5. **Flexible dependence**: Can model time-varying effects more naturally than splines
6. **Publication potential**: "Sparse Gaussian Processes for Multi-State Model Hazards"

### Concrete GP Use Cases

**Case 1: Epidemic modeling**
- Transmission rates vary smoothly over time
- Periodic kernels for seasonal patterns
- GP prior encodes epidemiologist's beliefs about rate changes

**Case 2: Medical device reliability**
- Hazard rates change with cumulative usage
- Non-stationary kernels for aging effects  
- Uncertainty critical for regulatory decisions

**Case 3: Customer churn**
- Complex temporal patterns in churn risk
- Can incorporate covariates in kernel
- Interpretable through kernel decomposition

---

## Next Steps if Pursuing GPs

1. **Proof of concept** (1 week)
   - Simple 2-state model with AbstractGPs.jl
   - Compare to exponential/Weibull baseline
   - Verify numerical integration accuracy

2. **Integration** (2 weeks)
   - Add GP option to hazard specification
   - Implement parameter updates via gradient descent
   - Test on real data

3. **Optimization** (1 week)
   - Benchmark vs splines
   - Optimize inducing point placement
   - Cache computations

4. **Validation** (2 weeks)
   - Simulation studies
   - Real data comparisons
   - Uncertainty calibration checks

5. **Documentation** (1 week)
   - Examples
   - Kernel selection guidance
   - Performance tuning tips

**Total effort: ~7 weeks for full implementation**

---

## Conclusion

**DataInterpolations.jl migration: Not recommended**
- Too much custom logic to reimplement
- Loses our sophisticated natural spline recombination
- Minimal benefit over current BSplineKit

**Sparse GPs: Highly recommended as future enhancement**
- Methodologically novel for multi-state models
- Natural fit for survival analysis
- Could be significant contribution to field
- Implement alongside splines, not as replacement

**Status quo: Perfectly fine**
- BSplineKit.jl is working great
- No urgent need to change
- Can revisit if specific limitations emerge
