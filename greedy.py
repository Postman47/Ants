import numpy as np

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

def greedy_cvrp(instance):
    """Implements a greedy algorithm for CVRP."""
    unvisited = list(instance.customers)
    routes = [[] for _ in range(instance.num_vehicles)]
    vehicle_index = 0
    total_visited = 0

    while unvisited:
        current_route = routes[vehicle_index]
        current_capacity = 0
        current_distance = 0
        last_location = 0  # Start from depot

        nearest_customer = None
        nearest_distance = float('inf')

        for customer in unvisited:
            distance = instance.distance_matrix[last_location, customer.id]
            if distance < nearest_distance:
                if current_capacity + customer.demand <= instance.vehicle_capacity and current_distance + distance <= instance.max_distance:
                    nearest_customer = customer
                    nearest_distance = distance

        if nearest_customer:
            current_route.append(nearest_customer.id)
            unvisited.remove(nearest_customer)
            current_capacity += nearest_customer.demand
            current_distance += nearest_distance
            last_location = nearest_customer.id
            total_visited +=1
        else:
            vehicle_index = (vehicle_index + 1) % instance.num_vehicles
            if not unvisited:
                break
            elif len(routes[vehicle_index]) != 0:
              routes[vehicle_index].append(0)

    for route in routes:
        if route and route[-1] != 0:
            route.append(0)

    if total_visited != len(instance.customers):
        print ("Warning: Not all customers visited")

    return Solution(routes)

# Example usage:
depot = Depot(0, 0)
customers = [Customer(1, 1, 5, 10), Customer(2, 2, 8, 15), Customer(3, 3, 2, 20), Customer(4, 4, 6, 12), Customer(5, 5, 9, 18)]
instance = CVRPInstance(depot, customers, num_vehicles=3, vehicle_capacity=50, max_distance=30)

solution = greedy_cvrp(instance)
distance = solution.calculate_total_distance(instance.distance_matrix)

print("Greedy Solution:", solution)
print("Greedy Distance:", distance)