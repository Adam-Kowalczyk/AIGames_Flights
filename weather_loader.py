import numpy as np
import pandas as pd
import datetime
#'%3a00%3a00'
def create_path(year, month, day, hour, prediction_hour, ending):
    path = './data/weather_and_winds/forecast'
    date =  datetime.datetime(year, month, day)
    hour_str = str(hour)
    if hour < 10:
        hour_str = '0' + hour_str
    path = path + str(prediction_hour) + '_' + date.strftime("%Y-%m-%d") + '_' + hour_str + ending + '.npz'
    return path


def get_weather_on_day(year, month, day, hour, prediction_hour):
    path = create_path(year, month, day, hour, prediction_hour, '%3a00%3a00')
    loaded = np.load(path)
    features = loaded.files
    return loaded

from scipy import spatial

def create_tree(lats, lngs):
    kdtree = spatial.KDTree(np.array(list(zip(lats.flatten(),lngs.flatten()))))
    return kdtree

def find_closest_spatial(lat,long,tree):
    return tree.query([(lat,long)])[1][0]

def convert_to_indices(index, width):
    ih = index % width
    iw = int(index / width)
    return iw, ih

def get_weather_pandas(loaded, row, column):
    data = {}
    for name in loaded.files:
        data[name] = [ loaded[name][row,column]]
    return pd.DataFrame(data)


