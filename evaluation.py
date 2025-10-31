import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def measure_distance(locations: pd.DataFrame, route: list | np.ndarray) -> float:
    """
    calculates the distance of the given route through the given travelling salesman problem
    :param locations: The x-y coordinates for each location in the TSP. Should be a 2d numpy array
    :param route: The route through the TSP. Should be a list or array of location indexes, in the desired order
                   The first entry should be the origin city. DO NOT include the origin city at the end of the list
                   too - the route will be assumed to return to the origin city after the list's last entry
    :return: the total distance of the specified route
    """
    assert len(set(route)) == locations.shape[0]

    # convert to numpy and tack the origin city onto the end of the route
    locations = locations.to_numpy()
    route = list(route) + [route[0]]

    # compute route distance
    start_pts = locations[route[:-1]]
    end_pts = locations[route[1:]]
    square_distances = ((end_pts - start_pts) ** 2).sum(axis=1)
    return (square_distances ** 0.5).sum()


def plot_route(locations: pd.DataFrame, route: list | np.ndarray) -> None:
    """
    plot the route through the TSP
    :param locations: The x-y coordinates for each location in the TSP. Should be a 2d numpy array
    :param route: The route through the TSP. Should be a list or array of location indexes, in the desired order
    :return: None
    """
    # scatterplot cities
    plt.scatter(locations['x'], locations['y'])

    # convert to numpy and tack the origin city onto the end of the route
    locations = locations.to_numpy()
    route = list(route) + [route[0]]

    # plot each leg of the route
    for i in range(len(route) - 1):
        plt.plot(locations[route[i:i+2], 0], locations[route[i:i+2], 1], 'k')
