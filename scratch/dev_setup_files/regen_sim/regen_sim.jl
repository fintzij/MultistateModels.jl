using ArraysOfArrays
using Chain
using DataFrames
using Distributions
using JLD2 # for saving files
using LinearAlgebra
using MultistateModels
using StatsBase
using Random

# load sim helpers
include(pwd()*"/scratch/dev_setup_files/regen_sim/sim_funs.jl");

seed = 1; cens = 2; nulleff = 1; jobid = 20001; sims_per_subj = 10; nboot = 10

# run the simulation
runtime = @elapsed results = work_function(seed, cens, nulleff, jobid, sims_per_subj, nboot)

@save "/data/fintzijr/multistate/sim2_regen/regen_sim_results_$jobid.jld2" results runtime