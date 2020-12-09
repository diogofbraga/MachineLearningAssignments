import csv
import time
import math
from statistics import mean


class KNN:
    def __init__(self, data, label, features, mode=1, n_neighbours=5, distance_function=1, radius=0, path=''):
        if path == '':
            self.data = data
        else:
            self.data = self.read_csv(path)
        self.label = label
        self.features = features
        self.mode = mode
        self.n_neighbours = n_neighbours
        self.distance_function = distance_function
        self.radius = radius

    def run(self, target):
        self.target = target
        neighbours = self.analyse()
        results = self.calculate(neighbours)
        return results

    def read_csv(self, path):
        read_file = open(path, newline='')
        data = csv.DictReader(read_file)
        return data

    def distance_calculation(self, row):
        if self.distance_function == 1: # Euclidean
            if len(self.features) == 1:
                distance = abs(float(row[self.features[0]])-float(self.target[self.features[0]]))
            elif len (self.features) > 1:
                features_sum = 0
                for x in self.features:
                    features_sum = features_sum + pow(float(row[x])-float(self.target[x]),2)
                distance = math.sqrt(features_sum)
        elif self.distance_function == 2: # Manhattan
            if len(self.features) == 1:
                distance = abs(float(row[self.features[0]])-float(self.target[self.features[0]]))
            elif len (self.features) > 1:
                features_sum = 0
                for x in self.features:
                    features_sum = features_sum + abs(float(row[x])-float(self.target[x]))
                distance = features_sum
        return distance

    def worst_neighbour(self, neighbours):
        worst_distance = -1
        for x in neighbours:
            if x['distance'] > worst_distance:
                worst_distance = x['distance']
                worst = x
        return worst

    def worst_distance_neighbour(self, neighbours):
        worst_distance = -1
        for x in neighbours:
            if x['distance'] > worst_distance:
                worst_distance = x['distance']
        return worst_distance

    def analyse(self):
        neighbours = []

        for row in self.data:
            #print(row) 
            distance = self.distance_calculation(row)
            if self.mode == 1: # KNeighbors
                worst_distance = self.worst_distance_neighbour(neighbours)
                if (len(neighbours) < self.n_neighbours) or (distance < worst_distance):
                    element = {'distance': distance, 'row': row}
                    if element['row'] != self.target:
                        neighbours.append(element)
                    if len(neighbours) > self.n_neighbours:
                        neighbours.remove(self.worst_neighbour(neighbours))
            elif self.mode == 2: # RadiusNeighbors
                if distance <= self.radius:
                    element = {'distance': distance, 'row': row}
                    if element['row'] != self.target:
                        neighbours.append(element)
        
        #print("Neighbours:")
        #for x in neighbours:
            #print("Distance:", x['distance'])
            #print("Row:", x['row'])

        return neighbours

    def calculate(self, neighbours):
        labels = []
        
        for x in neighbours:
            #print(x['row'][label])
            labels.append(float(x['row'][self.label]))
        
        result = mean(labels)
        return result