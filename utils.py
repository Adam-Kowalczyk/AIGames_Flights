import math

import numpy as np
from geopy import distance

def dist_in_miles_from_spherical_path(path): 
    dist = 0
    for idx, p2 in enumerate(path[1:]):
        dist += distance.distance(path[idx], p2).miles
    return dist

def spherical_to_3d_surface(vec):
    vec = (math.cos(math.radians(vec[1])) * math.cos(math.radians(vec[0])),
           math.sin(math.radians(vec[1])) * math.cos(math.radians(vec[0])),
           math.sin(math.radians(vec[0])))
    return np.array(vec)

def slerp_pos(start, end, t):
    start = spherical_to_3d_surface(start)
    end = spherical_to_3d_surface(end)
    theta = math.acos(np.dot(start, end))
    tween = (start * math.sin((1 - t) * theta) + end * math.sin(t * theta)) / math.sin(theta)
    lat = math.asin(tween[2])
    lng = math.acos(tween[0] / math.cos(lat))
    return [math.degrees(lat), -math.degrees(lng)]

def generate_slerp_path(path, points_num):
    dist_cum = [0]
    dist = 0

    for idx, p2 in enumerate(path[1:]):
        dist += distance.distance(path[idx], p2).miles
        dist_cum.append(dist)
    
    coords = [path[0]]
    to_id = 0
    for i in range(1, points_num+1):
        target = dist * i / points_num
        while target > dist_cum[to_id]:
            to_id += 1
        t = (target - dist_cum[to_id - 1]) / (dist_cum[to_id] - dist_cum[to_id - 1])
        coords.append(slerp_pos(path[to_id - 1], path[to_id], t))
    return coords

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = [[45.77736111111111, -111.15026111111112],
            [45.64619444444445, -110.80049722222222],
            [45.75552857271772, -110.70362717089138],
            [45.80855833333333, -108.62464722222222],
            [45.44118333333333, -104.64351388888889],
            [45.078175, -101.71507222222222],
            [44.49611111111111, -96.09027777777777],
            [44.00222222222222, -93.96111111111111],
            [43.18583333333333, -91.65916666666668],
            [42.38006111111111, -88.66791666666667],
            [41.97452222222223, -87.90659722222223]]
    slerp_path = generate_slerp_path(path, 20)

    route = np.asarray(path)
    fig, ax = plt.subplots()
    ax.scatter(route[:,0],route[:,1])
    for i in range(len(route[:,0])):
        ax.annotate(str(i), (route[:,0][i], route[:,1][i]))
        plt.scatter([x[0] for x in path], [x[1] for x in path], color = 'red')
    route2 = np.asarray(slerp_path)
    for i in range(len(route2[:,0])):
        ax.annotate(str(i), (route2[:,0][i], route2[:,1][i]))
        plt.scatter([x[0] for x in slerp_path], [x[1] for x in slerp_path], color = 'blue')
    plt.show()

    print(slerp_path)

