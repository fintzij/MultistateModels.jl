# set up a MultistateModel object
using Chain
using DataFrames
using MultistateModels

h12_ph = Hazard(@formula(0 ~ 1), "wei", 1, 2);
h21_ph = Hazard(@formula(0 ~ 1 + trt), "wei", 2, 1);

dat = 
    DataFrame(id = collect(1:3),
              tstart = [0, 0, 0],
              tstop = [10, 10, 10],
              statefrom = [1, 2, 1],
              stateto = [2,2,2],
              obstype = ones(3),
              trt = [0, 1, 0],
              age = [23, 32, 50])

# create multistate model object
msm_weiph = multistatemodel(h12_ph, h21_ph; data = dat)