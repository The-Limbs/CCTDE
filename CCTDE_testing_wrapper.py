from my_funcs import *
from CCTDE_core import calc_ccf


data = load_obj('example_input_data')
ts = data['time_series']


probe_dist = 10
correlation_threshold = 0.5

x1,y1 = (64,64)
x2,y2 = (64,64+probe_dist)
ts1 = ts[x1,y1,:]
ts2 = ts[x2,y2,:]

ccf,tau = calc_ccf(ts1,ts2,plot_bool=True)
