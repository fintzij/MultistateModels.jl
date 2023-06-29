using MultistateModels
using Distributions
using Plots

include("test/setup_ffbs.jl")


#
# Sampling when there is no censoring

# true trajectory
subj_dat_no_censoring.stateto

# emission matrix
subj_emat_no_censoring

# sampled trajectory
m, p = ForwardFiltering(subj_dat_no_censoring, tpm_book, subj_tpm_map, subj_emat_no_censoring) 
h = BackwardSampling(m, p) 

# verify match
all(subj_dat_no_censoring.stateto .== h)


#
# Impossible censoring patterns
subj_emat_impossible # impossible to have a transition from states (3,4) to states (1,2) at step 5 in a progressive model
sample_skeleton!(subj_dat_impossible, tpm_book, subj_tpm_map, subj_emat_impossible)


#
# comparing Monte Carlo estimates and analytical probabilities

# Monte Carlo
M = 100000
S = zeros(M)
for i in 1:M
    m, p = ForwardFiltering(subj_dat_analytical, tpm_book, subj_tpm_map, subj_emat_analytical) 
    h = BackwardSampling(m, p)
    S[i] = h[5]
end
sum(S.==4) / M

# analytical
1-exp(-1*2) # CDF of exponential distribution