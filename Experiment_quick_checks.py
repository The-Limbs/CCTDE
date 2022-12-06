from my_funcs import *
from CCTDE_core import calc_ccf,infer_1D_velocity

#load the data
data = load_obj('example_input_data')
ts = data['time_series']



px1_coor = (64,64)
px2_coor = (64,68)

print(np.linalg.norm(np.subtract(px1_coor,px2_coor)))
exit()

px1_ts = ts[px1_coor[0],px1_coor[1],:]
px2_ts = ts[px2_coor[0],px2_coor[1],:]

ccf,tau = calc_ccf(px1_ts,px2_ts)

correlation_threshold = 0.5
raw_velocity, vel_corr = infer_1D_velocity(ccf,tau,spatial_seperation,correlation_threshold)
print(raw_velocity)

