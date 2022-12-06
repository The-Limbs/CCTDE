from my_funcs import *
from CCTDE_core import calc_ccf,infer_1D_velocity,infer_2D_velocity
import 2to3

data = load_obj('example_input_data')
ts = data['time_series']


spatial_seperation = 10
correlation_threshold = 0.5

x1,y1 = (64,64)
x2,y2 = (64,64+spatial_seperation)
ts1 = ts[x1,y1,:]
ts2 = ts[x2,y2,:]

ccf,tau = calc_ccf(ts1,ts2)

velocity, maxcorr = infer_1D_velocity(ccf,tau,spatial_seperation,correlation_threshold)

# testing the find xy velocity routine
ref_location = (64,64)
velocities,correlations = infer_2D_velocity(ts,ref_location,spatial_seperation,correlation_threshold)
print(velocities)
