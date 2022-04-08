using MultistateModels

include("test/setup_2state_exp.jl")

# simulate a single path
MultistateModels.simulate_path(msm, 1)