# =============================================================================
# Smoothing Parameter Selection for Penalized Splines
# =============================================================================
#
# This file is a facade that includes the modularized smoothing selection code.
# The implementation is split across multiple files for maintainability:
#
#   smoothing_selection/
#   ├── header.jl           - Module header and documentation
#   ├── dispatch_exact.jl   - ExactData hyperparameter selection dispatch
#   ├── dispatch_markov.jl  - MPanelData dispatch, Markov state type, criteria
#   ├── dispatch_mcem.jl    - MCEMSelectionData dispatch, MCEM state type, criteria
#   ├── dispatch_general.jl - General nested optimization for ExactData
#   ├── common.jl           - SmoothingSelectionState, helper functions
#   ├── pijcv.jl            - PIJCV/CV criterion functions for ExactData
#   └── deprecated.jl       - Deprecated select_smoothing_parameters functions
#
# =============================================================================

# Include submodules in dependency order
include("smoothing_selection/header.jl")
include("smoothing_selection/dispatch_exact.jl")
include("smoothing_selection/dispatch_markov.jl")
include("smoothing_selection/dispatch_mcem.jl")
include("smoothing_selection/dispatch_general.jl")
include("smoothing_selection/common.jl")
include("smoothing_selection/pijcv.jl")
include("smoothing_selection/deprecated.jl")
