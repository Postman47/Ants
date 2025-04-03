import numpy as np
import time
import matplotlib.pyplot as plt
import os

from antColonySystem import ant_colony_system
from antSystem import ant_system
from basicAnts import ant_colony_optimization
from greedy import greedy_cvrp

from dataStructures import Depot, Solution, Customer, CVRPInstance


# (Customer, Depot, CVRPInstance, Solution, Ant, PheromoneMatrix, ant_colony_optimization, ant_system, ant_colony_system, greedy_cvrp)
# (Same implementations as in the previous responses)
# ... (Copy the code from the previous responses here) ...

def load_cvrp_instance(filepath):
    """Loads a CVRP instance from a file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    depot = None
    customers = []
    num_vehicles = 0
    vehicle_capacity = 0

    for line in lines:
        if "DIMENSION" in line:
            dimension = int(line.split(":")[1].strip())
        elif "CAPACITY" in line:
            vehicle_capacity = int(line.split(":")[1].strip())
        elif "NODE_COORD_SECTION" in line:
            coord_start = lines.index(line) + 1
        elif "DEMAND_SECTION" in line:
            demand_start = lines.index(line) + 1
        elif "VEHICLE" in line:
            num_vehicles = int(line.split(":")[1].strip())

    for i in range(coord_start, demand_start - 1):
        parts = lines[i].split()
        if int(parts[0]) == 1:
            depot = Depot(float(parts[1]), float(parts[2]))
        else:
            customers.append(Customer(int(parts[0]), float(parts[1]), float(parts[2]), 0))

    for i in range(demand_start, coord_start + dimension):
        parts = lines[i].split()
        for customer in customers:
            if customer.id == int(parts[0]):
                customer.demand = int(parts[1])

    return CVRPInstance(depot, customers, num_vehicles, vehicle_capacity, float('inf')) # max_distance is not in the benchmark file.

def run_experiment(instance, algorithm, params, num_runs=5):
    """Runs an experiment for a given algorithm and parameters."""
    results = []
    iterations_list = []
    times = []

    for _ in range(num_runs):
        start_time = time.time()
        if algorithm.__name__ == "greedy_cvrp":
            best_solution = algorithm(instance)
            best_distance = best_solution.calculate_total_distance(instance.distance_matrix)
            iterations = 1 # greedy has 1 iteration
        else:
            best_solution, best_distance = algorithm(instance, **params)
            iterations = params.get('max_iterations', 1)
        end_time = time.time()
        results.append(best_distance)
        iterations_list.append(iterations)
        times.append(end_time - start_time)

    return {
        'results': results,
        'iterations': iterations_list,
        'times': times,
        'avg_iterations': np.mean(iterations_list),
        'min_iterations': min(iterations_list),
        'max_iterations': max(iterations_list),
        'avg_distance': np.mean(results),
        'min_distance': min(results),
        'max_distance': max(results),
        'avg_time': np.mean(times)
    }

def visualize_results(results, instance, filename):
    """Visualizes the results of the experiments."""
    algorithms = list(results.keys())
    distances = [result['avg_distance'] for result in results.values()]
    iterations = [result['avg_iterations'] for result in results.values()]
    times = [result['avg_time'] for result in results.values()]
    stability = [result['results'] for result in results.values()]

    plt.figure(figsize=(15, 10))

    # Bar chart of average distances
    plt.subplot(2, 2, 1)
    plt.bar(algorithms, distances)
    plt.title('Average Solution Quality')
    plt.ylabel('Total Distance')

    # Bar chart of average iterations
    plt.subplot(2, 2, 2)
    plt.bar(algorithms, iterations)
    plt.title('Average Iterations')
    plt.ylabel('Iterations')

    # Bar chart of average execution times
    plt.subplot(2, 2, 3)
    plt.bar(algorithms, times)
    plt.title('Average Execution Time')
    plt.ylabel('Time (s)')

    # Box plot of solution stability
    plt.subplot(2, 2, 4)
    plt.boxplot(stability, labels=algorithms)
    plt.title('Solution Stability')
    plt.ylabel('Total Distance')

    plt.tight_layout()
    plt.savefig(f"{filename}_comparison.png")
    plt.close()

    # Route Visualization
    for algorithm_name, result in results.items():
        best_solution = Solution(result['results'].index(min(result['results'])) if algorithm_name != "greedy" else result['results'][0])
        visualize_routes(instance, best_solution, f"{filename}_{algorithm_name}_routes.png")

def visualize_routes(instance, solution, filename):
    """Visualizes the routes of a solution."""
    plt.figure(figsize=(10, 8))
    plt.scatter([instance.depot.x], [instance.depot.y], color='red', label='Depot')
    for customer in instance.customers:
        plt.scatter([customer.x], [customer.y], label=f'Customer {customer.id}')

    for route in solution.routes:
        if route:
            x_coords = [instance.depot.x] + [instance.customers[c - 1].x for c in route] + [instance.depot.x]
            y_coords = [instance.depot.y] + [instance.customers[c - 1].y for c in route] + [instance.depot.y]
            plt.plot(x_coords, y_coords, linestyle='-')

    plt.legend()
    plt.title('Route Visualization')
    plt.savefig(filename)
    plt.close()
# Example usage:
instance_path = "A-n32-k5.vrp"
instance = load_cvrp_instance(instance_path)

algorithms = {
    "basic_aco": (ant_colony_optimization, {'num_ants': 20, 'max_iterations': 100, 'alpha': 1, 'beta': 2, 'rho': 0.5, 'q': 100}),
    "ant_system": (ant_system, {'num_ants': 20, 'max_iterations': 100, 'alpha': 1, 'beta': 2, 'rho': 0.5, 'q': 100}),
    "ant_colony_system": (ant_colony_system, {'num_ants': 20, 'max_iterations': 100, 'alpha': 1, 'beta': 2, 'rho': 0.1, 'q': 100, 'q0': 0.9, 'tau_0': 0.1}),
    "greedy": (greedy_cvrp, {})
}

results = {}
for name, (algorithm, params) in algorithms.items():
    results[name] = run_experiment(instance, algorithm, params)
visualize_results(results, instance, "cvrp_results")