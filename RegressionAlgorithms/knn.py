import csv
import time
import math
from statistics import mean, sqrt

def read_data(path):
    read_file = open(path, newline='')
    data = csv.DictReader(read_file)
    return data

def distance_calculation(row):
    if distance_function == 1: # Euclidean
        if len(features) == 1:
            distance = abs(float(row[features[0]])-float(target[features[0]]))
        elif len (features) > 1:
            features_sum = 0
            for x in features:
                features_sum = features_sum + pow(float(row[x])-float(target[x]),2)
            distance = sqrt(features_sum)
    elif distance_function == 2: # Manhattan
        if len(features) == 1:
            distance = abs(float(row[features[0]])-float(target[features[0]]))
        elif len (features) > 1:
            features_sum = 0
            for x in features:
                features_sum = features_sum + abs(float(row[x])-float(target[x]))
            distance = features_sum
    return distance
        
def analyse(data, n_neighours):
    neighbours = []

    for row in data:
        #print(row) 
        distance = distance_calculation(row)
        if mode == 1:
            if (len(neighbours) < n_neighours) or (distance <= neighbours[0]['distance']):
                best = {'distance': distance, 'row': row}
                neighbours.insert(0, best)
                if len(neighbours) > n_neighours:
                    neighbours.pop()
        elif mode == 2:
            if distance <= radius:
                best = {'distance': distance, 'row': row}
                neighbours.append(best)
    
    print("Neighbours:")
    for x in neighbours:
        print("Distance:", x['distance'])
        print("Row:", x['row'])

    return neighbours

def calculate(neighbours):
    labels = []
    
    for x in neighbours:
        print(x['row'][label])
        labels.append(float(x['row'][label]))
    
    result = mean(labels)
    return result

# Global variables
path = '../MetroInterstateTrafficVolume/MetroInterstateTrafficVolume.csv'
mode = 1 # 1 = KNeighbors; 2 = RadiusNeighbors
n_neighours = 5
distance_function = 1 # 1 = Euclidean Distance; 2 = Manhattan Distance
radius = 0 # 0 indicates no radius
label = 'traffic_volume'
features = ['temp']
target = {'holiday': 'None', 'temp': '276.42', 'rain_1h': '0.0', \
        'snow_1h': '0.0', 'clouds_all': '20', 'weather_main': 'Haze', \
        'weather_description': 'haze', 'date_time': '2016-02-19 00:00:00', 'traffic_volume': '708'}

if __name__ == '__main__':
    start = time.time()
    data = read_data(path)
    neighbours = analyse(data, n_neighours)
    result = calculate(neighbours)
    print("Result:", result)
    end = time.time()
    print("Time:", end-start)