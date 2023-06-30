# set up a MultistateModel object
using DataFrames
using ExponentialUtilities
using Distributions
using MultistateModels
using MultistateModels: draw_samplepath, SampleSkeleton!, build_tpm_mapping, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!, ForwardFiltering, BackwardSampling


n_obs = 5
dat = DataFrame(id        = fill(1, n_obs),
                tstart    = [1, 2, 5, 6, 8],
                tstop     = [2, 5, 6, 8, 9],
                statefrom = [1, 1, 2, 2, 3],
                stateto   = [1, 2, 2, 3, 4],
                obstype   = fill(2, n_obs)) # temporary values for obstype
 
censoring_patterns = [[3 1 0 0 0]; # only one state is allowed in each censoring pattern
                      [4 0 1 0 0];
                      [5 0 0 1 0];
                      [6 0 0 0 1]]
# progressive model
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
h34 = Hazard(@formula(0 ~ 1), "exp", 3, 4)                     
model = multistatemodel(h12, h23, h34; data = dat, censoring_patterns = censoring_patterns)

set_parameters!(
    model, 
    (h12 = [log(1)],
     h23 = [log(1.5)],
     h34 = [log(1.75)]))
parameters = flatview(model.parameters)

# set up tpm, etc
books = build_tpm_mapping(model.data) 
pars = model.parameters
hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
tpm_book = build_tpm_book(Float64, model.tmat, books[1])
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
for t in eachindex(books[1])
    compute_hazmat!(hazmat_book[t],pars,model.hazards,books[1][t])
    compute_tmat!(tpm_book[t],hazmat_book[t],books[1][t],cache)
end
tpm_map = books[2]
subj=1
subj_inds = model.subjectindices[subj]
subj_dat     = view(model.data, subj_inds, :)
subj_tpm_map = view(tpm_map   , subj_inds, :)


#
# no censoring
censoring_patterns = [[3 1 0 0 0]; # only one state is allowed in each censoring pattern
                      [4 0 1 0 0];
                      [5 0 0 1 0];
                      [6 0 0 0 1]]
obstype_no_censoring = [3,3,4,5,5] # match the true trajectory
dat.obstype = obstype_no_censoring
model_no_censoring = multistatemodel(h12, h23, h34; data = dat, censoring_patterns = censoring_patterns)
subj_emat_no_censoring = view(model_no_censoring.emat, subj_inds, :)
subj_dat_no_censoring = view(model_no_censoring.data, subj_inds, :)


#
# impossible censoring patterns
censoring_patterns = [3 1 1 0 0;
                      4 0 1 1 0;
                      5 0 0 1 1]
obstype_impossible = [3, 3, 4, 5, 3] # at step 5, impossible to go from states (3,4) to states (1,2) in this progressive model
dat.obstype = obstype_impossible
model_impossible = multistatemodel(h12, h23, h34; data = dat, censoring_patterns = censoring_patterns)
subj_emat_impossible = view(model_impossible.emat, subj_inds, :)
subj_dat_impossible = view(model_impossible.data, subj_inds, :)


#
# Monte Carlo
censoring_patterns = [3 0 1 1 0] # uncertainty between states 2 and 3
obstype_analytical = [2, 3, 3, 3, 2] # states in steps 2-4 are censored 
dat.obstype = obstype_analytical
model_analytical = multistatemodel(h12, h23, h34; data = dat, censoring_patterns = censoring_patterns)
subj_emat_analytical = view(model_analytical.emat, subj_inds, :)
subj_dat_analytical = view(model_analytical.data, subj_inds, :)