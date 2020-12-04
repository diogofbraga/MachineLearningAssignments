import csv
import time
import math
from statistics import mean, sqrt

def read_data(path):
    read_file = open(path, newline='')
    data = csv.DictReader(read_file)
    return data
        
def analyse(data, n_neighours):
    shortest = 1000
    neighbours = []
    
    for row in data:
        #print(row)
        if distance_function == 1:
            if len(features) == 1:
                distance = abs(float(row[features[0]])-float(target[features[0]]))
            elif len (features) > 1:
                features_sum = 0
                for x in features:
                    features_sum = features_sum + pow(float(row[x])-float(target[x]),2)
                distance = sqrt(features_sum)
        elif distance_function == 2:
            if len(features) == 1:
                distance = abs(float(row[features[0]])-float(target[features[0]]))
            elif len (features) > 1:
                features_sum = 0
                for x in features:
                    features_sum = features_sum + abs(float(row[x])-float(target[x]))
                distance = features_sum
        if distance <= shortest:
            shortest = distance
            best = {'distance': shortest, 'row': row}
            neighbours.insert(0, best)
            if len(neighbours) > n_neighours:
                neighbours.pop()
            #print(shortest)
    
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
n_neighours = 5
distance_function = 1 # 1 = Euclidean Distance; 2 = Manhattan Distance
label = 'traffic_volume'
features = ['temp', 'clouds_all', 'rain_1h', 'snow_1h']
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