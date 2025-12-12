# MultistateModels

Monte-Carlo expectation-maximization algorithm for fitting multi-state semi-Markov models to panel data in `Julia`.
The algorithm can fit parametric and semi-parametric transition intensities with a proportional intensity parameterization for covariate effects.

## Installation

You can install the development version of `MultistateModels` from
[GitHub](https://github.com/) with:

```
using Pkg
Pkg.add(url="https://github.com/fintzij/MultistateModels.jl.git")
```

## Testing

Run the test suite:

```julia
using Pkg
Pkg.test("MultistateModels")
```

By default, quick unit tests run (~2 min). For the full suite including statistical validation tests:

```bash
MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
```

### Extended Test Suite

For comprehensive stress testing and statistical validation, see the separate test package:
**[MultistateModelsTests.jl](https://github.com/fintzij/MultistateModelsTests.jl)**

To use:
```julia
# Clone into this directory
cd /path/to/MultistateModels.jl
git clone https://github.com/fintzij/MultistateModelsTests.jl MultistateModelsTests

# Activate and run
cd MultistateModelsTests
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using MultistateModelsTests; MultistateModelsTests.run_longtests()'
```
