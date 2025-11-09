using DataFrames
using LinearAlgebra
using MultistateModels
using Random
using Test

# set seed
Random.seed!(52787)

# setup file for generating model objects 
# include("setup.jl") # this could be included in the individual .jl files
include("setup_3state_expwei.jl")
include("setup_3state_weiph.jl")
include("setup_2state_trans.jl")
include("setup_gompertz.jl")
# include("setup_splines.jl")  # TODO: Splines not yet implemented in infrastructure_changes
include("test_miscellaneous.jl")

@testset "runtests" begin
    include("test_modelgeneration.jl")
    include("test_hazards.jl")
    include("test_helpers.jl")
    include("test_make_subjdat.jl")
    # include("test_loglik.jl")  # TODO: Tests old API that no longer exists - needs complete rewrite for new likelihood infrastructure
end
