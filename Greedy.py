import math
from dataReader import *

class Vehicle:
    """Represents a delivery vehicle with a capacity and max distance constraint."""
    def __init__(self, capacity, max_distance):
        self.capacity = capacity
        self.current_load = 0
        self.max_distance = max_distance

    def can_accommodate(self, demand):
        """Checks if the vehicle can accommodate the given demand."""
        return self.current_load + demand <= self.capacity

    def load(self, demand):
        """Loads the vehicle with the given demand."""
        if self.can_accommodate(demand):
            self.current_load += demand
            return True
        return False

    def __repr__(self):
        return f"Vehicle(capacity={self.capacity}, current_load={self.current_load}, max_distance={self.max_distance})"

class Route:
    """Represents a single vehicle's delivery route."""
    def __init__(self, depot, max_distance):
        self.depot = depot
        self.customers = []
        self.total_demand = 0
        self.current_distance = 0
        self.max_distance = max_distance
        self.max_capacity = None # Will be set when the route is associated with a vehicle

    def add_customer(self, customer, distance_from_previous):
        """Adds a customer to the route and updates route information."""
        self.customers.append(customer)
        self.total_demand += customer.demand
        self.current_distance += distance_from_previous

    def calculate_total_distance(self, customer_table):
        """Calculates the total distance of the route (including return to depot)."""
        if not self.customers:
            return 0
        distance = self.distance(self.depot, self.customers[0])
        for i in range(len(self.customers) - 1):
            distance += self.distance(self.customers[i], self.customers[i+1])
        distance += self.distance(self.customers[-1], self.depot)
        return distance

    def can_add(self, customer, customer_table):
        """Checks if adding a customer violates capacity or max distance constraints."""
        if self.max_capacity is None or self.total_demand + customer.demand > self.max_capacity:
            return False

        last_node = self.customers[-1] if self.customers else self.depot
        distance_to_customer = self.distance(last_node, customer)
        distance_back_to_depot = self.distance(customer, self.depot)

        temp_distance = self.current_distance + distance_to_customer
        # Estimate total route distance if this customer is added
        if self.customers:
            # Subtract the distance of the current return trip and add the new one
            current_return = self.distance(self.customers[-1], self.depot)
            potential_total_distance = (self.calculate_total_distance(customer_table) - current_return) + distance_to_customer + distance_back_to_depot
        else:
            potential_total_distance = self.distance(self.depot, customer) + distance_back_to_depot

        return potential_total_distance <= self.max_distance

    def distance(self, node1, node2):
        """Calculates the Euclidean distance between two nodes."""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def __repr__(self):
        return f"Route(customers={[c.id for c in self.customers]}, demand={self.total_demand}, distance={self.calculate_total_distance(None):.2f})"

class Solution:
    """Represents a complete CVRP solution with multiple routes."""
    def __init__(self, depot, capacity, max_distance):
        self.depot = depot
        self.capacity = capacity
        self.max_distance = max_distance
        self.routes = []

    def add_route(self, route):
        """Adds a route to the solution."""
        self.routes.append(route)

    def calculate_total_distance(self):
        """Calculates the total distance of the entire solution."""
        return sum(route.calculate_total_distance(None) for route in self.routes)

    def is_valid(self, customer_table):
        """Checks if the solution respects capacity and max distance constraints and serves all customers."""
        served_customers = set()
        for route in self.routes:
            if route.total_demand > self.capacity:
                return False
            if route.calculate_total_distance(customer_table) > self.max_distance:
                return False
            for customer in route.customers:
                served_customers.add(customer.id)
        return served_customers == set(customer_table.keys())

    def __repr__(self):
        return f"Solution(num_routes={len(self.routes)}, total_distance={self.calculate_total_distance():.2f})"

def calculate_distance(node1, node2):
    """Calculates the Euclidean distance between two nodes."""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def is_solution_valid(solution, customer_table, capacity, max_distance):
    """Checks if a given solution is valid (including max distance)."""
    served_customers = set()
    for route in solution.routes:
        if route.total_demand > capacity:
            return False
        if route.calculate_total_distance(customer_table) > max_distance:
            return False
        for customer in route.customers:
            served_customers.add(customer.id)
    return served_customers == set(customer_table.keys())

def greedy_algorithm(depot, customer_table, capacity, max_distance):
    """Implements a greedy algorithm for the CVRP with max distance constraint."""
    solution = Solution(depot, capacity, max_distance)
    unvisited_customers = set(customer_table.values())

    while unvisited_customers:
        current_vehicle = Vehicle(capacity, max_distance)
        current_route = Route(depot, max_distance)
        current_route.max_capacity = capacity
        current_location = depot

        while True:
            nearest_customer = None
            min_distance = float('inf')

            for customer in unvisited_customers:
                distance = calculate_distance(current_location, customer)
                if current_vehicle.can_accommodate(customer.demand) and current_route.can_add(customer, customer_table) and distance < min_distance:
                    min_distance = distance
                    nearest_customer = customer

            if nearest_customer:
                current_route.add_customer(nearest_customer, min_distance)
                current_vehicle.load(nearest_customer.demand)
                unvisited_customers.remove(nearest_customer)
                current_location = nearest_customer
            else:
                # No more customers can be added to the current route
                if current_route.customers:
                    current_route.current_distance += calculate_distance(current_location, depot)
                    solution.add_route(current_route)
                break

        # If there are still unvisited customers but no route could be formed, it might indicate an issue
        # or a limitation of this simple greedy approach.
        if unvisited_customers and not solution.routes[-1].customers if solution.routes else unvisited_customers:
            # Try starting a new route from the depot to the nearest unvisited customer
            nearest_remaining = None
            min_dist_remaining = float('inf')
            temp_route = Route(depot, max_distance)
            temp_vehicle = Vehicle(capacity, max_distance)

            for cust in unvisited_customers:
                if temp_vehicle.can_accommodate(cust.demand) and temp_route.can_add(cust, customer_table):
                    dist = calculate_distance(depot, cust)
                    if dist < min_dist_remaining:
                        min_dist_remaining = dist
                        nearest_remaining = cust

            if nearest_remaining:
                new_route = Route(depot, max_distance)
                new_route.max_capacity = capacity
                new_vehicle = Vehicle(capacity, max_distance)
                distance_to = calculate_distance(depot, nearest_remaining)
                distance_back = calculate_distance(nearest_remaining, depot)
                if distance_to + distance_back <= max_distance and new_vehicle.can_accommodate(nearest_remaining.demand):
                    new_route.add_customer(nearest_remaining, distance_to)
                    new_vehicle.load(nearest_remaining.demand)
                    new_route.current_distance += distance_back
                    solution.add_route(new_route)
                    unvisited_customers.remove(nearest_remaining)
                else:
                    break # Cannot even serve the nearest remaining customer within max distance
            else:
                break # No remaining customer can be the start of a new valid route

    return solution, solution.calculate_total_distance()

if __name__ == '__main__':
    file_path = '/home/piotr/PycharmProjects/ants/Ants/data/Vrp-Set-E/E/E-n22-k4.vrp'
    number_of_trucks, optimal_value, capacity, depot, customer_table = read_vrp_file(file_path)
    max_distance = 300 # Example max distance, adjust as needed

    best_solution, best_distance = greedy_algorithm(depot, customer_table, capacity, max_distance)

    print("\n--- Greedy Algorithm Solution with Max Distance ---")
    if best_solution:
        print(f"Total distance: {best_distance:.2f}")
        for i, route in enumerate(best_solution.routes):
            print(f"Route {i + 1}: {route}")
            print(f"  Route Total Distance: {route.calculate_total_distance(customer_table):.2f}")
        print(f"Is solution valid: {is_solution_valid(best_solution, customer_table, capacity, max_distance)}")
    else:
        print("No valid solution found.")