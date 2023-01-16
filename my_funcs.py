import pickle
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# save object using pickle
def save_obj(obj,name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)

# load object using pickle
def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f,encoding = 'latin1')
