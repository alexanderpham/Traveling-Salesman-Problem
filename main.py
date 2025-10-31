"""
COMP 3625 ASG 1: Travelling Salesman

Authors: Ethan Ai, Alex Pham, Felix Yap
Prof: Eric Chalmers
Date: 10/06/2024
=====================================================================

Description:
------------
This project implements a solution to the Travelling Salesman Problem (TSP) using a hybrid algorithm that combines
a Minimum Spanning Tree (MST) with a Nearest Neighbor (NN) search. The algorithm aims to find a good approximation
of the shortest possible route that visits each city exactly once and returns to the starting point.

The algorithm works by:
1. Calculating the Euclidean distance matrix between cities.
2. Constructing an MST using Prim's algorithm to guide the selection of city connections.
3. Dynamically adjusting a rate (alpha) that controls the influence of MST (structural connections) vs NN (distance-based) during the decision process.
4. Randomly choosing 100 starting cities and selecting the best route based on the total route distance.

Modules and Functions:
-----------------------
1. calculate_distance_matrix(locations):
   - Computes the Euclidean distance between each pair of cities.

2. construct_mst(num_cities, dist_matrix):
   - Constructs a Minimum Spanning Tree (MST) using Prim's algorithm to form a sparse structure of city connections.

3. select_next_city(current_city, unvisited_cities, mst_neighbors, dist_matrix, alpha):
   - Selects the next city based on a weighted combination of MST and Nearest Neighbor influence.

4. nn_and_mst_tsp(locations, dist_matrix, mst_edges, start_city):
   - Performs a hybrid search that starts with more influence from the MST and progressively favors the NN algorithm as cities are visited.

5. find_route(locations):
   - Randomly selects 100 different starting cities calling with the nn_and_mst_tsp function and returns the best route found.

"""

import pandas as pd
import evaluation
import time
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import random

def calculate_distance_matrix(locations):
    """
    calculates the distance of a Eucliden matrix for cities
    takes a 2D array and its coordinates 
    returns a 2D array with a matrix that is the distance
    """
    num_cities = len(locations)
    dist_matrix = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = np.sqrt((locations[i, 0] - locations[j, 0])**2 + (locations[i, 1] - locations[j, 1])**2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist  # Distance is symmetric
    
    return dist_matrix

def construct_mst(num_cities, dist_matrix):
    """
    builds a MST using Prim's algorithm.
    :param num_cities: The total number of cities.
    :param dist_matrix: The computed distance matrix.
    :return: List of edges in the MST.
    """
    
    
    in_mst = [False] * num_cities
    mst_edges = []
    
    pq = PriorityQueue()
    
    # city 0 (cost, city index, parent city index)
    # Parent is -1 (no parent for first node)
    pq.put((0, 0, -1))  

  
    while len(mst_edges) < num_cities - 1:
        # takes the city with the min cost based on distance
        cost, current, parent = pq.get()

        if in_mst[current]:
            continue
        
        in_mst[current] = True
        
        #if not the startinc city, add edge to MST
        if parent != -1:
            mst_edges.append((parent, current))

        #check neighboring cities
        for neighbor in range(num_cities):
            #if neigbhor hasn't been added to tree in MST then add to queue
            if not in_mst[neighbor]:
                pq.put((dist_matrix[current][neighbor], neighbor, current))

    return mst_edges
    
def select_next_city(current_city, unvisited_cities, mst_neighbors, dist_matrix, alpha):
    """
    Selects the next city to visit based on a weighted combination of MST and Nearest Neighbor.
    
    :param current_city: The current city that is expanded.
    :param unvisited_cities: List of cities that have not yet been visited.
    :param mst_neighbors: List of cities connected to the current city via the MST.
    :param dist_matrix: Distance matrix for all city pairs.
    :param alpha: Rate controlling the influence of MST vs NN.
    :return: The next city to visit.
    """
    next_city = None
    min_score = float('inf')

    for city in unvisited_cities:

        dist_score = dist_matrix[current_city][city] # score is city between current city and unvistied
        mst_bonus = -alpha * dist_score if city in mst_neighbors else 0 #bonus MST if neighbors 
        
        #calc the learning value alpha to see which city to go to next
        score = (1 - alpha) * dist_score + mst_bonus

        # Debugging output
        # if city in mst_neighbors:
        #     print(f"City {city} is connected by MST with bonus: {mst_bonus :.8f} and score: {score :.6f}")
        # else:
        #     print(f"City {city} is not connected by MST, score: {score :.6f}")

        # Choose the city with the best (lowest) score
        if score < min_score:
            min_score = score
            next_city = city

    print(f"Selected next city: {next_city} with score: {min_score :.6f}")
    return next_city

def nn_and_mst_tsp(locations, dist_matrix, mst_edges, start_city):
    """
    Uses Nearest Neighbor algorithm while using MST to prioritize city selections.
    A dynamic rate (alpha) is used to balance between MST and NN influences.
    The alpha value will be larger at the start of the search (favoring MST) and smaller towards the
    end of the search (favoring NN).
    
    :param locations: DataFrame of city coordinates.
    :param dist_matrix: Distance matrix.
    :param mst_edges: The MST edges for city selection.
    :param start_city: The starting city for the nearest neighbor algorithm.
    :param initial_alpha: Initial rate that controls the weight of MST influence.
    :return: A list representing the TSP route.
    """
    initial_alpha = 0.6 
    num_cities = len(locations)
    
    current_city = start_city
    route = [current_city]
    visited = set([current_city])

    #adjacent list from MST edges of neighbors 
    adj_list = {i: [] for i in range(num_cities)}
    for u, v in mst_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    print(f"Starting at city {start_city}, with initial rate: {initial_alpha:.6f}")
    
    while len(visited) < num_cities:
        alpha = initial_alpha * (1 - len(visited) / num_cities) #decrease alpha with the number of cities that get visited 
        unvisited_cities = list(set(range(num_cities)) - visited)
        mst_neighbors = list(set(adj_list[current_city]) - visited)
        next_city = select_next_city(current_city, unvisited_cities, mst_neighbors, dist_matrix, alpha)

        # Print to show choice
        print(f"Moving from city {current_city} to city {next_city} with dynamic rate: {alpha:.6f}")

        visited.add(next_city)
        route.append(next_city)
        current_city = next_city

    return route

def find_route(locations: pd.DataFrame) -> list:
    """
    Find route function that calls the tsp algorithm. This function will choose 100 unique start locations
    and choose the best route.
    :param locations: The x-y coordinates for each location in the TSP. Should be a pandas DataFrame.
    :return: The route through the TSP as a list of location indexes.
    """
    num_cities = len(locations)
    locations_array = locations.to_numpy()
    
    # Compute the distance matrix
    dist_matrix = calculate_distance_matrix(locations_array)
    
    # Compute the MST
    mst_edges = construct_mst(num_cities, dist_matrix)
    
    # Find an optimal route
    best_route = None
    best_distance = float('inf')
    
    # Generate 100 unique starting cities or the all of the cities if total is less than 100
    starting_points = random.sample(range(num_cities), min(100, num_cities))

    for start_city in starting_points:
        route = nn_and_mst_tsp(locations, dist_matrix, mst_edges, start_city)
        route_distance = evaluation.measure_distance(locations, route)

        if route_distance < best_distance:
            best_distance = route_distance
            best_route = route

    return best_route

if __name__ == '__main__':

    tsp = pd.read_csv('./data/250a.csv', index_col=0)
    start_time = time.time()
    
    route = find_route(tsp)
    
    # Measure the elapsed time
    elapsed = time.time() - start_time

    distance = evaluation.measure_distance(tsp, route)
    print(f'found a route with distance {distance:.2f} in {elapsed:.4f}s')

    # Plot route
    evaluation.plot_route(tsp, route)
    plt.title(f'distance={distance:.2f}')
    plt.xticks([])
    plt.yticks([])
    plt.show()
