# MultistateModels.jl JSS Paper - Document Plan

## Target Journal
Journal of Statistical Software (JSS) - Software paper format

## Overview
Comprehensive documentation of MultistateModels.jl implementing and extending methods from Morsomme et al. (2025) for fitting semi-Markov multistate models to panel data via Monte Carlo EM.

---

## Document Structure

### 1. Abstract (~250 words)
- Package purpose: Fitting continuous-time multistate models to exact and panel data
- Key innovation: MCEM algorithm enabling semi-Markov models with interval-censored observations
- Supported hazard families: Exponential, Weibull, Gompertz, B-spline, Phase-type (Coxian)
- Covariate support: Proportional hazards, time-varying covariates
- Variance estimation: Model-based, infinitesimal jackknife, jackknife
- Implementation language: Julia

### 2. Introduction (~1,500 words)
#### 2.1 Motivation
- Multistate models in clinical research (disease progression, treatment response)
- Challenge: Panel data with interval-censored transition times
- Limitation of Markov assumption for many applications
- Need for semi-Markov models with flexible sojourn distributions

#### 2.2 Existing Software Landscape
**R Packages:**
| Package | Key Features | Limitations |
|---------|-------------|-------------|
| msm (Jackson 2011) | Markov + HMM, panel data, covariates | Markov only (no semi-Markov for panel) |
| mstate (Putter et al. 2007) | Non-parametric, Aalen-Johansen | Requires exact event times |
| SemiMarkov (Krol & Saint-Pierre 2015) | Semi-Markov, Weibull | **Requires exact transition times** |
| flexsurv (Jackson 2016) | Parametric, splines, semi-Markov (clock-reset) | Requires exact transition times |
| icmstate (Gomon & Putter 2024) | Non-parametric interval-censored | Non-parametric only |

**Key Gap:** No existing software fits parametric/semi-parametric semi-Markov models to panel data.

#### 2.3 Package Contributions
1. Unified interface for Markov and semi-Markov multistate models
2. MCEM algorithm for semi-Markov models with panel data
3. Flexible hazard specifications (parametric, spline, phase-type)
4. Multiple variance estimation methods
5. Efficient Julia implementation with automatic differentiation

#### 2.4 Paper Organization

### 3. Mathematical Framework (~3,000 words)
#### 3.1 Multistate Processes
- State space S = {1, ..., K}, absorbing states
- Transition intensities/hazards: λ_{jk}(t | H_t)
- Sojourn times and holding time distributions

#### 3.2 Markov vs Semi-Markov Models
- Markov property: Future depends only on current state
- Semi-Markov: Transition rates depend on time in current state
- Clock reset at each transition

#### 3.3 Cause-Specific Hazards
- Definition: λ_{jk}(t) = lim_{h→0} P(transition to k in (t, t+h] | in state j at t) / h
- Relationship to transition probabilities
- Competing risks interpretation

#### 3.4 Parametric Hazard Families
**Exponential:** λ(t) = exp(β₀)
**Weibull:** λ(t) = (κ/σ)(t/σ)^{κ-1}, log-parameterization
**Gompertz:** λ(t) = exp(β₀ + β₁t)
**B-Spline:** λ(t) = exp(B(t)′β) with I-spline basis for monotonicity

#### 3.5 Phase-Type Distributions
- Coxian representation of sojourn distributions
- Latent Markov chain on phases
- Absorption into next observable state
- Flexibility to approximate any sojourn distribution

#### 3.6 Covariate Effects
- Proportional hazards: λ_{jk}(t | x) = λ_{jk,0}(t) exp(β′x)
- Time-varying covariates: Piecewise-constant intensity approach
- Accelerated failure time interpretation (future)

#### 3.7 Likelihood Formulations
**Exact data:** Direct product of hazard contributions
$$\mathcal{L}(\theta) = \prod_{i=1}^{n} \prod_{m=1}^{M_i} \lambda_{j_m k_m}(t_m) \exp\left(-\int_0^{t_m} \sum_{k \neq j_m} \lambda_{j_m k}(u) du\right)$$

**Panel data (Markov):** Matrix exponential likelihood
$$\mathcal{L}(\theta) = \prod_{i=1}^{n} \prod_{m=1}^{M_i} P_{j_m k_m}(t_{m-1}, t_m; \theta)$$

**Panel data (Semi-Markov):** Requires marginalization over latent paths

### 4. Model Construction (~2,500 words)
#### 4.1 The Hazard() Constructor
- Syntax overview
- Family specification: :exp, :wei, :gom, :sp, :pt
- Transition specification: statefrom, stateto
- Optional formula for covariates

#### 4.2 Spline Hazards
- Degree, knots, boundary knots
- Natural splines and extrapolation
- Monotonicity constraints (monotone = -1, 0, 1)

#### 4.3 Phase-Type Hazards
- n_phases specification
- Coxian structure constraints
- State expansion mechanics

#### 4.4 The multistatemodel() Function
- Combining hazard specifications
- Data requirements and format
- Surrogate specification for MCEM
- Model validation

#### 4.5 Data Format
- Required columns: id, tstart, tstop, statefrom, stateto, obstype
- Observation types: 1 (exact), 2 (panel), 3 (censored)
- Covariate columns
- Multiple data streams

#### 4.6 Initial Values and Constraints
- Parameter initialization strategies
- Box constraints via lower/upper bounds

### 5. Simulation (~1,500 words)
#### 5.1 Path Simulation Algorithm
- Gillespie-style exact simulation
- Handling competing risks
- Time-varying covariates in simulation

#### 5.2 The simulate() Function
- Basic usage
- Generating paths vs. panel observations
- Multiple realizations
- Setting random seeds

#### 5.3 Simulation for Model Validation
- Posterior predictive checks
- Coverage studies

### 6. Inference (~4,500 words)
#### 6.1 The fit() Function Interface
- Unified interface dispatching to appropriate method
- Key arguments: maxiter, tol, verbose, compute_vcov

#### 6.2 Exact Data: Direct Maximum Likelihood
- Separable likelihood structure
- Transition-specific optimization
- Automatic differentiation with ForwardDiff.jl

#### 6.3 Panel Data with Markov Models
- Matrix exponential computation
- Uniformization for numerical stability
- Direct optimization

#### 6.4 Panel Data with Semi-Markov Models: MCEM Algorithm
##### 6.4.1 Algorithm Overview
- E-step: Sample latent paths given parameters
- M-step: Maximize expected complete-data log-likelihood
- Convergence monitoring

##### 6.4.2 Importance Sampling for Path Generation
- Proposal distribution construction
- Markov surrogate proposals
- Phase-type proposals for better approximation
- Weight computation and normalization

##### 6.4.3 Forward-Filtering Backward-Sampling (FFBS)
- Forward pass: Compute filtering distributions
- Backward pass: Sample state sequence
- Uniformization for continuous-time sampling

##### 6.4.4 Sampling Importance Resampling (SIR)
- Initial pool generation
- Pareto-smoothed importance sampling (PSIS) diagnostics
- Latin hypercube resampling for diversity
- Adaptive pool sizing

##### 6.4.5 Effective Sample Size and Adaptation
- ESS computation from importance weights
- Automatic ESS targeting
- Iteration-adaptive sample sizes

##### 6.4.6 SQUAREM Acceleration
- Squared iterative methods for EM acceleration
- Step length selection
- Monotonicity enforcement

##### 6.4.7 Convergence Diagnostics
- Parameter change monitoring
- Marginal likelihood estimation
- Monte Carlo error assessment

#### 6.5 Proposal Distributions
##### 6.5.1 MarkovProposal
- Uses Markov surrogate
- Efficient for near-Markov processes
- Weight computation

##### 6.5.2 PhaseTypeProposal
- Phase-type expanded state space
- Better approximation of semi-Markov dynamics
- Computational trade-offs

#### 6.6 Optimization Details
- Optimizer selection (LBFGS, Newton)
- Gradient computation via AD
- Handling constraints

#### 6.7 Variance Estimation
##### 6.7.1 Model-Based (Observed Fisher Information)
- Hessian computation at MLE
- Louis' formula for MCEM
- Computation via automatic differentiation

##### 6.7.2 Infinitesimal Jackknife (IJ/Sandwich)
- Robust variance estimation
- Score contributions from paths
- Implementation details
- Interpretation as sandwich estimator

##### 6.7.3 Jackknife Variance Estimation
- Leave-one-out resampling
- Computational considerations
- Comparison with IJ estimator

##### 6.7.4 Variance Estimator Selection Guidelines

### 7. Model Selection and Regularization (~1,000 words) [PLACEHOLDERS]
#### 7.1 Information Criteria
- AIC, BIC computation
- Effective parameters in MCEM context

#### 7.2 Cross-Validation (Planned)
- K-fold CV for multistate models
- Blocked CV for correlated observations
- Implementation roadmap

#### 7.3 Penalized Splines (Planned)
- P-spline formulation
- Penalty selection
- Integration with MCEM

#### 7.4 Neural ODE Hazards (Planned)
- Universal differential equations
- Neural network hazard functions
- Training considerations

### 8. Computational Details (~2,000 words)
#### 8.1 Julia Implementation
- Language choice rationale
- Key dependencies: DifferentialEquations.jl, ForwardDiff.jl, Optim.jl

#### 8.2 Memory Management
- Pre-allocated workspaces
- Path storage strategies
- Garbage collection considerations

#### 8.3 Numerical Stability
- Log-space computations
- Underflow prevention
- Matrix exponential computation

#### 8.4 Parallel Computing
- Thread-safe path sampling
- Subject-level parallelism
- Future: Distributed computing

#### 8.5 Performance Benchmarks
- Comparison with R packages
- Scaling with subjects and states

### 9. Model Outputs and Diagnostics (~1,500 words)
#### 9.1 Fitted Model Objects
- Parameter extraction: get_parameters(), get_parameters_flat()
- Variance-covariance: get_vcov()
- Confidence intervals

#### 9.2 Convergence Diagnostics
- MCEM trace plots
- ESS monitoring
- Pareto-k diagnostics for importance sampling
- Monte Carlo standard errors

#### 9.3 Model Checking
- Residual analysis
- Observed vs. expected transitions
- Posterior predictive checks

### 10. Examples (~3,500 words)
#### 10.1 Illness-Death Model (Basic)
- Three-state progressive model
- Exact observation data
- Exponential hazards
- Full workflow demonstration

#### 10.2 Panel Data with Weibull Hazards
- Interval-censored observations
- Semi-Markov sojourn times
- MCEM fitting
- Variance estimation comparison

#### 10.3 Reversible Model with Covariates
- Bidirectional transitions
- Treatment effects
- Time-varying covariates
- Spline hazards

#### 10.4 Phase-Type Sojourn Distributions
- Flexible sojourn modeling
- Comparison with parametric
- Diagnostic checks

#### 10.5 Multi-Stream Data Integration
- Combining exact and panel observations
- Different observation schedules
- Clinical trial application

### 11. Comparison with Existing Software (~1,500 words)
#### 11.1 Feature Comparison Table

| Feature | MultistateModels.jl | msm | mstate | SemiMarkov | flexsurv |
|---------|---------------------|-----|--------|------------|----------|
| Panel data | ✓ | ✓ | ✗ | ✗ | ✗ |
| Semi-Markov | ✓ | ✗ | ✗ | ✓ | ✓ |
| **Semi-Markov + Panel** | **✓** | **✗** | **✗** | **✗** | **✗** |
| Spline hazards | ✓ | ✗ | ✗ | ✗ | ✓ |
| Phase-type | ✓ | ✗ | ✗ | ✗ | ✗ |
| Time-varying covariates | ✓ | ✓ | ✓ | ✓ | ✓ |
| Variance: IJ/Sandwich | ✓ | ✗ | ✗ | ✗ | ✗ |
| Variance: Jackknife | ✓ | ✓ | ✗ | ✗ | ✗ |

#### 11.2 Methodological Comparisons
- msm: Markov assumption limits applicability
- SemiMarkov: Cannot handle panel data
- mstate/flexsurv: Require exact times

#### 11.3 Performance Comparisons
- Computational benchmarks
- Memory usage
- Ease of use

#### 11.4 When to Use Each Package

### 12. Summary and Future Directions (~800 words)
#### 12.1 Summary of Contributions
#### 12.2 Current Limitations
#### 12.3 Planned Extensions
- Penalized splines with automatic smoothing
- Cross-validation framework
- Neural ODE hazard functions
- Bayesian inference via HMC
- GPU acceleration

---

## Appendices

### Appendix A: Uniformization Algorithm
- Mathematical derivation
- Implementation details
- Numerical considerations

### Appendix B: FFBS for Semi-Markov Models
- Complete algorithm specification
- Pseudo-code

### Appendix C: Phase-Type Distribution Theory
- Coxian distributions
- Matrix representations
- Closure properties

### Appendix D: Variance Estimation Derivations
- Louis' formula for MCEM
- Infinitesimal jackknife derivation
- Sandwich estimator connection

### Appendix E: Proofs and Technical Details

---

## References (Key Citations)

### Methodology
1. Morsomme R, Xu J, Schiemsky T, Beck A, Roychoudhury S, Fintz J (2025). "Assessing treatment efficacy for interval-censored endpoints using multistate semi-Markov models fit to multiple data streams." arXiv:2501.14097v3.

2. Alaa AM, van der Schaar M (2018). "A Hidden Absorbing Semi-Markov Model for Informatively Censored Temporal Data: Learning and Inference." Journal of Machine Learning Research, 19(4):1-62.

3. Wei GCG, Tanner MA (1990). "A Monte Carlo Implementation of the EM Algorithm and the Poor Man's Data Augmentation Algorithms." Journal of the American Statistical Association, 85(411):699-704.

4. Varadhan R, Roland C (2008). "Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm." Scandinavian Journal of Statistics, 35(2):335-353.

5. Vehtari A, Gelman A, Gabry J (2017). "Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC." Statistics and Computing, 27(5):1413-1432.

### Phase-Type Distributions
6. Neuts MF (1981). Matrix-Geometric Solutions in Stochastic Models. Johns Hopkins University Press.

7. Asmussen S, Nerman O, Olsson M (1996). "Fitting Phase-Type Distributions via the EM Algorithm." Scandinavian Journal of Statistics, 23(4):419-441.

8. Bladt M, Nielsen BF (2017). Matrix-Exponential Distributions in Applied Probability. Springer.

### Software
9. Jackson CH (2011). "Multi-State Models for Panel Data: The msm Package for R." Journal of Statistical Software, 38(8):1-28.

10. Putter H, Fiocco M, Geskus RB (2007). "Tutorial in Biostatistics: Competing Risks and Multi-State Models." Statistics in Medicine, 26(11):2389-2430.

11. Krol A, Saint-Pierre P (2015). "SemiMarkov: An R Package for Parametric Estimation in Multi-State Semi-Markov Models." Journal of Statistical Software, 66(6):1-16.

12. Jackson CH (2016). "flexsurv: A Platform for Parametric Survival Modeling in R." Journal of Statistical Software, 70(8):1-33.

13. Gomon D, Putter H (2024). "Nonparametric estimation of interval-censored multi-state models." arXiv:2409.07176.

### Variance Estimation
14. Louis TA (1982). "Finding the Observed Information Matrix When Using the EM Algorithm." Journal of the Royal Statistical Society B, 44(2):226-233.

15. Efron B, Hinkley DV (1978). "Assessing the Accuracy of the Maximum Likelihood Estimator: Observed Versus Expected Fisher Information." Biometrika, 65(3):457-483.

16. Jaeckel LA (1972). "The Infinitesimal Jackknife." Bell Laboratories Memorandum.

### Numerical Methods
17. Sidje RB (1998). "Expokit: A Software Package for Computing Matrix Exponentials." ACM Transactions on Mathematical Software, 24(1):130-156.

18. Jensen A (1953). "Markoff Chains as an Aid in the Study of Markoff Processes." Scandinavian Actuarial Journal, 1953(sup1):87-91.

### Julia Ecosystem
19. Bezanson J, Edelman A, Karpinski S, Shah VB (2017). "Julia: A Fresh Approach to Numerical Computing." SIAM Review, 59(1):65-98.

20. Revels J, Lubin M, Papamarkou T (2016). "Forward-Mode Automatic Differentiation in Julia." arXiv:1607.07892.

---

## Estimated Length
- Main text: ~24,000 words
- Appendices: ~6,000 words  
- Total: ~30,000 words (~60-70 pages)

## File Structure
```
docs/jss_paper/
├── jss_document_plan.md          (this file)
├── MultistateModels_JSS.qmd      (main document)
├── references.bib                 (BibTeX references)
├── figures/
│   ├── illness_death_diagram.pdf
│   ├── mcem_convergence.pdf
│   ├── variance_comparison.pdf
│   └── ...
├── code/
│   ├── example_illness_death.jl
│   ├── example_panel_weibull.jl
│   ├── example_phasetype.jl
│   └── ...
└── _extensions/
    └── jss/                       (JSS Quarto extension)
```

## Next Steps
1. Create BibTeX reference file
2. Set up Quarto document with JSS formatting
3. Draft sections in order: 3 → 4 → 5 → 6 → 10 → 11 → 2 → 7-9 → 12 → Appendices
