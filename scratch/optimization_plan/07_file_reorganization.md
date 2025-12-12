# File Reorganization Plan

## Overview

This document outlines the plan for reorganizing the MultistateModels.jl source files
from a flat structure to a purpose-based modular structure.

## Current Structure (24,711 lines total)

| File | Lines | Purpose |
|------|-------|---------|
| phasetype.jl | 4,598 | Phase-type distribution fitting, surrogates |
| crossvalidation.jl | 2,432 | Nested cross-validation |
| likelihoods.jl | 2,099 | Likelihood computation |
| helpers.jl | 1,965 | Parameter handling, flatten/unflatten |
| sampling.jl | 1,821 | Path sampling, FFBS |
| hazards.jl | 1,762 | Hazard evaluation |
| modelfitting.jl | 1,683 | fit() function |
| common.jl | 1,504 | Type definitions |
| simulation.jl | 1,289 | Path simulation |
| mcem.jl | 966 | MCEM algorithm |
| modelgeneration.jl | 889 | Model construction |
| miscellaneous.jl | 665 | Utility functions |
| statsutils.jl | 541 | Statistical utilities |
| smooths.jl | 427 | Spline utilities |
| modeloutput.jl | 398 | Display/printing |
| initialization.jl | 393 | Parameter initialization |
| surrogates.jl | 364 | Surrogate model utilities |
| macros.jl | 350 | @hazard macro |
| pathfunctions.jl | 328 | Path manipulation |

## Target Structure (by purpose)

```
src/
├── types/           # All type definitions
│   ├── types.jl     # Main include
│   ├── hazards.jl   # Abstract + concrete hazard types
│   ├── surrogates.jl
│   ├── models.jl
│   ├── data.jl
│   ├── configuration.jl
│   └── utilities.jl
│
├── construction/    # Model/hazard building
│   ├── construction.jl
│   ├── modelgeneration.jl
│   └── macros.jl
│
├── evaluation/      # Hazard evaluation, likelihoods
│   ├── evaluation.jl
│   ├── hazards.jl
│   └── likelihoods.jl
│
├── inference/       # Model fitting
│   ├── inference.jl
│   ├── modelfitting.jl
│   ├── mcem.jl
│   └── crossvalidation.jl
│
├── simulation/      # Path simulation, sampling
│   ├── simulation.jl
│   └── sampling.jl
│
├── accessors/       # Parameter access, display
│   ├── accessors.jl
│   ├── parameters.jl
│   └── modeloutput.jl
│
└── utilities/       # Helper functions
    ├── utilities.jl
    ├── helpers.jl
    ├── statsutils.jl
    └── miscellaneous.jl
```

## Dependency Graph

The main dependency challenge is circular references. Here's the dependency order:

1. **types/** (no dependencies on other modules)
   - Abstract types first
   - Then concrete types that use the abstracts

2. **utilities/** (depends only on types)
   - Flatten/unflatten
   - Statistical helpers

3. **construction/** (depends on types, utilities)
   - Model building
   - Hazard construction

4. **evaluation/** (depends on types, utilities, construction)
   - Hazard evaluation
   - Likelihood computation

5. **simulation/** (depends on types, utilities, evaluation)
   - Path simulation
   - Sampling algorithms

6. **inference/** (depends on all above)
   - Model fitting
   - MCEM
   - Cross-validation

7. **accessors/** (depends on types, possibly all)
   - Parameter get/set
   - Display functions

## Migration Strategy

### Phase 1: Type Consolidation
- Move all type definitions from common.jl to types/ directory
- Keep common.jl as a thin include that just re-exports

### Phase 2: Utility Extraction
- Move flatten/unflatten code to utilities/
- Move statistical helpers to utilities/

### Phase 3: Evaluation Module
- Extract hazard evaluation code
- Extract likelihood code

### Phase 4: Construction Module
- Move modelgeneration.jl
- Move macros.jl

### Phase 5: Simulation Module
- Move simulation.jl
- Move sampling.jl

### Phase 6: Inference Module
- Move modelfitting.jl
- Move mcem.jl
- Move crossvalidation.jl

### Phase 7: Accessors Module
- Move parameter accessors
- Move modeloutput.jl

## Key Challenges

1. **Circular Dependencies**: Many types reference each other
2. **Forward Declarations**: Julia doesn't support forward declarations
3. **Incremental Testing**: Need to test at each step
4. **Backward Compatibility**: Keep exports working

## Files Created (not yet integrated)

The following files have been created but are not yet included in the main module:

- src/types/types.jl (main include)
- src/types/hazards.jl (hazard types)
- src/types/surrogates.jl (surrogate types)
- src/types/models.jl (model types)
- src/types/data.jl (data types)
- src/types/configuration.jl (config types)
- src/types/utilities.jl (utility types)

These contain type definitions extracted from common.jl but need:
1. Dependency resolution (imports from main module)
2. Integration testing
3. Update to main module includes

## Status

- [x] Directory structure created
- [x] Type files created (draft)
- [ ] Type files tested
- [ ] Main module updated
- [ ] Tests passing
