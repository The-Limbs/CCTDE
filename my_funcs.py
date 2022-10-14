import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

matplotlib.rc('image', interpolation='none', origin='lower', cmap = 'seismic'\
              ,aspect=1.3)
cmap = matplotlib.cm.seismic
cmap.set_bad(color='k')
matplotlib.rcParams['mathtext.fontset']='stix'
matplotlib.rcParams['font.family']='STIXGeneral'
matplotlib.rcParams['font.size']=26
matplotlib.rcParams['text.usetex']=True



# save object using pickle
def save_obj(obj,name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)

# load object using pickle
def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
