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

jobid = 96; seed = 96; cens = 1; nulleff = 1;  sims_per_subj = 20; nboot = 1000

# run the simulation
runtime = @elapsed results = work_function(seed, cens, nulleff, jobid, sims_per_subj, nboot)

@save "/data/fintzijr/multistate/sim2_regen/regen_sim_results_$jobid.jld2" results runtime