# ===============================================================
# Variance Estimation Module
# ===============================================================
# 
# This file serves as the main entry point for variance estimation
# functionality. The implementation is split into focused submodules:
# 
# - gradient_hessian.jl: Core gradient and Hessian computation for Exact/Markov
# - fisher_mcem.jl: Fisher information for MCEM
# - ij_variance.jl: IJ/sandwich/robust variance estimation
# - pijcv.jl: PIJCV (Preconditioned IJC Variance) implementation
# - constrained.jl: Constrained variance estimation
# 
# ===============================================================

include("variance/gradient_hessian.jl")
include("variance/fisher_mcem.jl")
include("variance/ij_variance.jl")
include("variance/pijcv.jl")
include("variance/constrained.jl")
