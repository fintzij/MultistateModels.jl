using DataFrames
using ExponentialUtilities
using Distributions
using Plots
using MultistateModels
using MultistateModels: draw_samplepath, SampleSkeleton!, build_tpm_mapping, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!, ForwardFiltering, BackwardSampling, flatview, MPanelData, loglik


# progressive four-state model
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
h34 = Hazard(@formula(0 ~ 1), "exp", 3, 4)

# parameters
pars=(h12 = [log(1)], h23 = [log(1.5)], h34 = [log(1.75)])

# small dataset with 1 individual and 5 observations
times = [1, 2, 5, 6, 8, 9]
n_obs = 5
dat = DataFrame(id        = fill(1, n_obs),
                tstart    = times[1:end-1],
                tstop     = times[2:end],
                statefrom = fill(999, n_obs), # temporary values
                stateto   = fill(999, n_obs), # temporary values
                obstype   = fill(999, n_obs)) # temporary values
                  
#
# Censoring with degenerate censoring patterns (only one state allowed)
CensoringPatterns_degenerate = [3 1 0 0 0;
                                4 0 1 0 0;        
                                5 0 0 1 0;
                                6 0 0 0 1]

# set up data
dat.obstype = [3,3,4,5,5] # censoring patterns
dat.statefrom = [1, 0, 0, 0, 0]
dat.stateto = fill(0, n_obs)

# set up model
model_degenerate = multistatemodel(h12, h23, h34; data = dat, CensoringPatterns = CensoringPatterns_degenerate);
set_parameters!(model_degenerate, pars)

# construct subj_tpm_map and tpm_book
books_degenerate = build_tpm_mapping(model_degenerate.data)
tpm_book_degenerate = build_tpm_book(eltype(flatview(model_degenerate.parameters)), model_degenerate.tmat, books_degenerate[1])
hazmat_book_degenerate = build_hazmat_book(eltype(flatview(model_degenerate.parameters)), model_degenerate.tmat, books_degenerate[1])
cache_degenerate = ExponentialUtilities.alloc_mem(similar(hazmat_book_degenerate[1]), ExpMethodGeneric())
for t in eachindex(books_degenerate[1])
    compute_hazmat!(hazmat_book_degenerate[t],model_degenerate.parameters,model_degenerate.hazards,books_degenerate[1][t])
    compute_tmat!(tpm_book_degenerate[t],hazmat_book_degenerate[t],books_degenerate[1][t],cache_degenerate)
end


#
# Simple setup for comparing FFBS estimates and Monte Carlo estimates
CensoringPatterns_MC = [3 0 1 1 0] # uncertainty between states 2 and 3
dat.obstype = [2,3,3,3,2] # steps 2-4 are state-censored 
dat.statefrom = [1, 0, 0, 0, 0]
dat.stateto = [1, 0, 0, 0, 4]
model_MC = multistatemodel(h12, h23, h34; data = dat, CensoringPatterns = CensoringPatterns_MC);
set_parameters!(model_MC, pars)

books_MC = build_tpm_mapping(model_MC.data)
tpm_book_MC = build_tpm_book(eltype(flatview(model_MC.parameters)), model_MC.tmat, books_MC[1])
hazmat_book_MC = build_hazmat_book(eltype(flatview(model_MC.parameters)), model_MC.tmat, books_MC[1])
cache_MC = ExponentialUtilities.alloc_mem(similar(hazmat_book_MC[1]), ExpMethodGeneric())
for t in eachindex(books_MC[1])
    compute_hazmat!(hazmat_book_MC[t],model_MC.parameters,model_MC.hazards,books_MC[1][t])
    compute_tmat!(tpm_book_MC[t],hazmat_book_MC[t],books_MC[1][t],cache_MC)
end


#
# Censoring with an impossible trajectory
CensoringPatterns_impossible = [3 1 1 0 0; 
                                4 0 1 1 0;
                                5 0 0 1 1]
dat.obstype = [3,3,4,5,3] # at step 5, it is impossible to go from states (3,4) to states (1,2) in this progressive model
dat.statefrom = [1, 0, 0, 0, 0]
dat.stateto = fill(0, n_obs)
model_impossible = multistatemodel(h12, h23, h34; data = dat, CensoringPatterns = CensoringPatterns_impossible);
set_parameters!(model_impossible, pars)

books_impossible = build_tpm_mapping(model_impossible.data)
tpm_book_impossible = build_tpm_book(eltype(flatview(model_impossible.parameters)), model_impossible.tmat, books_impossible[1])
hazmat_book_impossible = build_hazmat_book(eltype(flatview(model_impossible.parameters)), model_impossible.tmat, books_impossible[1])
cache_impossible = ExponentialUtilities.alloc_mem(similar(hazmat_book_impossible[1]), ExpMethodGeneric())
for t in eachindex(books_impossible[1])
    compute_hazmat!(hazmat_book_impossible[t],model_impossible.parameters,model_impossible.hazards,books_impossible[1][t])
    compute_tmat!(tpm_book_impossible[t],hazmat_book_impossible[t],books_impossible[1][t],cache_impossible)
end