import numpy as np
import random

class Customer:
    """Represents a customer with demand and coordinates."""
    def __init__(self, id, x, y, demand):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand

    def __repr__(self):
        return f"Customer(id={self.id}, demand={self.demand})"

class Depot:
    """Represents the depot with coordinates."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Depot"

class CVRPInstance:
    """Represents the Capacitated Vehicle Routing Problem instance."""
    def __init__(self, depot, customers, num_vehicles, vehicle_capacity, max_distance):
        self.depot = depot
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.max_distance = max_distance
        self.distance_matrix = self.calculate_distance_matrix()

    def calculate_distance_matrix(self):
        """Calculates the distance matrix between all locations (depot and customers)."""
        locations = [self.depot] + self.customers
        num_locations = len(locations)
        distance_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(num_locations):
                distance_matrix[i, j] = np.sqrt((locations[i].x - locations[j].x)**2 + (locations[i].y - locations[j].y)**2)
        return distance_matrix

    def get_demand(self, customer_id):
        """Gets the demand of a customer by its ID."""
        for customer in self.customers:
            if customer.id == customer_id:
                return customer.demand
        return 0  # Return 0 if customer not found

    def __repr__(self):
        return f"CVRPInstance(num_vehicles={self.num_vehicles}, num_customers={len(self.customers)})"

class Solution:
    """Represents a solution to the CVRP, a list of routes."""
    def __init__(self, routes):
        self.routes = routes

    def calculate_total_distance(self, distance_matrix):
        """Calculates the total distance of the solution."""
        total_distance = 0
        for route in self.routes:
            for i in range(len(route) - 1):
                total_distance += distance_matrix[route[i], route[i + 1]]
            if route:
                total_distance += distance_matrix[route[-1], 0] # back to depot
        return total_distance

    def is_feasible(self, instance):
        """Checks if the solution is feasible (capacity and distance constraints)."""
        for route in self.routes:
            if not route:
                continue
            capacity = 0
            distance = 0
            for i in range(len(route)):
                if i > 0:
                    distance += instance.distance_matrix[route[i-1], route[i]]
                capacity += instance.get_demand(route[i]) if route[i] != 0 else 0
            distance += instance.distance_matrix[route[-1], 0] # back to depot
            if capacity > instance.vehicle_capacity or distance > instance.max_distance:
                return False
        return True

    def __repr__(self):
        return f"Solution(routes={self.routes})"

class Ant:
    """Represents an ant that constructs a solution."""
    def __init__(self, instance, q0):
        self.instance = instance
        self.visited = [False] * (len(instance.customers) + 1) # +1 for depot
        self.current_location = 0  # Start at the depot
        self.route = []
        self.routes = []
        self.capacity = 0
        self.distance = 0
        self.current_route = []
        self.q0 = q0

    def reset(self):
        """Resets the ant for a new solution construction."""
        self.visited = [False] * (len(self.instance.customers) + 1)
        self.current_location = 0
        self.route = []
        self.routes = []
        self.capacity = 0
        self.distance = 0
        self.current_route = []

    def choose_next_customer(self, pheromone_matrix, alpha, beta):
        """Chooses the next customer to visit based on pheromone and heuristic information (ACS)."""
        unvisited = []
        for i, customer in enumerate(self.instance.customers):
            if not self.visited[customer.id]:
                unvisited.append(customer)

        if not unvisited:
            return 0  # Go back to depot

        probabilities = []
        for customer in unvisited:
            distance = self.instance.distance_matrix[self.current_location, customer.id]
            pheromone = pheromone_matrix[self.current_location, customer.id]
            heuristic = 1 / distance if distance > 0 else 0.0
            probabilities.append((pheromone ** alpha) * (heuristic ** beta))

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        if random.random() < self.q0:
            # Exploitation: choose the best customer
            best_customer_index = np.argmax(probabilities)
            return unvisited[best_customer_index].id
        else:
            # Exploration: choose a customer based on probabilities
            chosen_customer = np.random.choice(unvisited, p=probabilities)
            return chosen_customer.id

    def move_to(self, customer_id, pheromone_matrix, tau_0):
        """Moves the ant to the given customer and updates its state (ACS)."""
        if self.current_location != customer_id:
            pheromone_matrix[self.current_location, customer_id] = (1 - tau_0) * pheromone_matrix[self.current_location, customer_id] + tau_0 * 1
        self.visited[self.current_location] = True
        if customer_id == 0:
            self.routes.append(self.current_route[:])
            self.current_route = []
            self.capacity = 0
            self.distance = 0
        else:
            self.capacity += self.instance.get_demand(customer_id)
            self.distance += self.instance.distance_matrix[self.current_location, customer_id]
            self.current_route.append(customer_id)
        self.current_location = customer_id

    def construct_solution(self, pheromone_matrix, alpha, beta, tau_0):
        """Constructs a solution by visiting customers (ACS)."""
        self.reset()
        vehicle_count = 0;

        while vehicle_count < self.instance.num_vehicles and (False in self.visited[1:]):
            next_customer = self.choose_next_customer(pheromone_matrix, alpha, beta)
            if next_customer == 0 or self.capacity + self.instance.get_demand(next_customer) > self.instance.vehicle_capacity or self.distance + self.instance.distance_matrix[self.current_location, next_customer] > self.instance.max_distance:
                if self.current_route:
                    self.move_to(0,pheromone_matrix,tau_0)
                    vehicle_count +=1
                else:
                    break;
            else:
                self.move_to(next_customer,pheromone_matrix,tau_0)

        if self.current_route:
            self.move_to(0,pheromone_matrix,tau_0)

        return Solution(self.routes)

class PheromoneMatrix:
    """Manages the pheromone matrix."""
    def __init__(self, instance, initial_pheromone=1.0):
        self.matrix = np.full(instance.distance_matrix.shape, initial_pheromone)

    def update_pheromone(self, best_solution, instance, evaporation_rate, q):
        """Updates the pheromone matrix based on the best solution (ACS)."""
        self.matrix *= (1 - evaporation_rate)  # Evaporate all pheromones first
        if best_solution and best_solution.is_feasible(instance):
            distance = best_solution.calculate_total_distance(instance.distance_matrix)
            if distance > 0:
                delta_pheromone = q / distance
                for route in best_solution.routes:
                    for i in range(len(route) - 1):
                        self.matrix[route[i], route[i + 1]] += delta_pheromone
                    if route:
                        self.matrix[route[-1], 0] += delta_pheromone

def ant_colony_system(instance, num_ants, max_iterations, alpha, beta, rho, q, q0, tau_0):
    """Implements the Ant Colony System algorithm for CVRP."""
    pheromone_matrix = PheromoneMatrix(instance)
    best_solution = None
    best_distance = float('inf')
    global_best_solution = None
    global_best_distance = float('inf')

    for iteration in range(max_iterations):
        solutions = []
        for _ in range(num_ants):
            ant = Ant(instance,q0)
            solution = ant.construct_solution(pheromone_matrix.matrix, alpha, beta, tau_0)
            solutions.append(solution)
        current_best_solution = None
        current_best_distance = float('inf')

        for solution in solutions:
            if solution.is_feasible(instance):
                distance = solution.calculate_total_distance(instance.distance_matrix)
                if distance < current_best_distance:
                    current_best_distance = distance
                    current_best_solution = solution
                if distance < global_best_distance:
                    global_best_distance = distance
                    global_best_solution = solution

        pheromone_matrix.update_pheromone(global_best_solution, instance, rho, q)

    return global_best_solution, global_best_distance

# Example usage:
depot = Depot(0, 0)
customers = [Customer(1, 1, 5, 10), Customer(2, 2, 8, 15), Customer(3, 3, 2, 20), Customer(4, 4, 6, 12), Customer(5, 5, 9, 18)]
instance = CVRPInstance(depot, customers, num_vehicles=3, vehicle_capacity=50, max_distance=30)

best_solution, best_distance = ant_colony_system(instance, num_ants=10, max_iterations=100, alpha=1, beta=2, rho=0.1, q=100, q0=0.9, tau_0 = 0.1)

print("Best Solution:", best_solution)
print("Best Distance:", best_distance)