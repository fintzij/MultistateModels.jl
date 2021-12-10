using MultistateModels, DataFrames

#### Minimal model
h12 = Hazard(@formula(lambda12 ~ trt), "exp", 1, 2);
h23 = Hazard(@formula(lambda21 ~ trt), "wei", 2, 3);
h13 = Hazard(@formula(lambda13 ~ trt), "exp", 1, 3);

hazards = (h12, h23, h13)