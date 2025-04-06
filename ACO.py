import math
import random
import sys

from dataReader import *

class Vehicle:
    """Represents a delivery vehicle with a capacity constraint."""
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
        if self.total_demand + customer.demand > self.max_capacity:
            return False

        last_node = self.customers[-1] if self.customers else self.depot
        distance_to_customer = self.distance(last_node, customer)
        distance_back_to_depot = self.distance(customer, self.depot)

        temp_distance = self.current_distance + distance_to_customer + distance_back_to_depot
        if self.customers:
            # Need to subtract the previous return trip and add the new one
            distance_without_last_return = self.calculate_total_distance(customer_table) - self.distance(self.customers[-1], self.depot)
            temp_distance = distance_without_last_return + distance_to_customer + distance_back_to_depot
        elif not self.customers:
            temp_distance = self.distance(self.depot, customer) + self.distance(customer, self.depot)


        return temp_distance <= self.max_distance

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
    """Calculates the Euclidean distance between two nodes (can be Depot or Customer)."""
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

class Ant:
    """Represents an ant that constructs a CVRP solution."""
    def __init__(self, id, num_customers, capacity, depot, max_distance):
        self.id = id
        self.num_customers = num_customers
        self.capacity = capacity
        self.depot = depot
        self.max_distance = max_distance
        self.visited_customers = [False] * num_customers
        self.solution = Solution(depot, capacity, max_distance)
        self.current_route = Route(depot, max_distance)
        self.current_route.max_capacity = capacity # Store max capacity in the route as well
        self.solution.add_route(self.current_route)
        self.current_vehicle = Vehicle(capacity, max_distance)

    def reset(self):
        """Resets the ant for a new solution construction."""
        self.visited_customers = [False] * self.num_customers
        self.solution = Solution(self.depot, self.capacity, self.max_distance)
        self.current_route = Route(self.depot, self.max_distance)
        self.current_route.max_capacity = self.capacity
        self.solution.add_route(self.current_route)
        self.current_vehicle = Vehicle(self.capacity, self.max_distance)

    def can_visit(self, customer, customer_table):
        """Checks if the ant can visit a customer (not visited, within capacity, and within max distance)."""
        if self.visited_customers[customer.id - 1]:
            return False
        if not self.current_vehicle.can_accommodate(customer.demand):
            return False

        # Check if adding the customer violates max distance
        last_node = self.current_route.customers[-1] if self.current_route.customers else self.depot
        distance_to_customer = calculate_distance(last_node, customer)
        distance_back_to_depot = calculate_distance(customer, self.depot)

        potential_route_distance = self.current_route.current_distance + distance_to_customer
        # If the route is not empty, we need to consider the return trip from the previous last customer
        if self.current_route.customers:
            distance_without_last_return = self.current_route.calculate_total_distance(customer_table) - calculate_distance(self.current_route.customers[-1], self.depot)
            potential_total_distance = distance_without_last_return + distance_to_customer + distance_back_to_depot
        else:
            potential_total_distance = calculate_distance(self.depot, customer) + calculate_distance(customer, self.depot)

        return potential_total_distance <= self.max_distance

    def select_next_customer(self, pheromone_matrix, customer_table, alpha, beta):
        """Selects the next customer to visit based on pheromone and heuristic, considering max distance."""
        unvisited_feasible_customers = [
            cust for cust_id, cust in customer_table.items()
            if not self.visited_customers[cust_id - 1] and self.current_vehicle.can_accommodate(cust.demand)
        ]

        if not unvisited_feasible_customers:
            return None

        probabilities = {}
        current_node = self.current_route.customers[-1] if self.current_route.customers else self.depot

        total_probability = 0
        for customer in unvisited_feasible_customers:
            # Check if adding this customer violates max distance
            last_node = current_node
            distance_to_customer = calculate_distance(last_node, customer)
            distance_back_to_depot = calculate_distance(customer, self.depot)

            potential_route_distance = self.current_route.current_distance + distance_to_customer
            if self.current_route.customers:
                distance_without_last_return = self.current_route.calculate_total_distance(customer_table) - calculate_distance(self.current_route.customers[-1], self.depot)
                potential_total_distance = distance_without_last_return + distance_to_customer + distance_back_to_depot
            else:
                potential_total_distance = calculate_distance(self.depot, customer) + calculate_distance(customer, self.depot)

            if potential_total_distance <= self.max_distance:
                pheromone = pheromone_matrix[current_node.id if isinstance(current_node, Customer) else 0][customer.id]
                distance = calculate_distance(current_node, customer)
                heuristic = 1 / distance if distance > 0 else float('inf')
                probability = (pheromone ** alpha) * (heuristic ** beta)
                probabilities[customer.id] = probability
                total_probability += probability

        if total_probability == 0:
            # Fallback: randomly choose an unvisited and feasible customer that doesn't violate max distance
            valid_next_customers = []
            for customer in unvisited_feasible_customers:
                last_node = current_node
                distance_to_customer = calculate_distance(last_node, customer)
                distance_back_to_depot = calculate_distance(customer, self.depot)
                if self.current_route.customers:
                    distance_without_last_return = self.current_route.calculate_total_distance(customer_table) - calculate_distance(self.current_route.customers[-1], self.depot)
                    potential_total_distance = distance_without_last_return + distance_to_customer + distance_back_to_depot
                else:
                    potential_total_distance = calculate_distance(self.depot, customer) + calculate_distance(customer, self.depot)
                if potential_total_distance <= self.max_distance:
                    valid_next_customers.append(customer)

            return random.choice(valid_next_customers) if valid_next_customers else None

        # Roulette wheel selection
        random_value = random.uniform(0, total_probability)
        cumulative_probability = 0
        for customer_id, probability in probabilities.items():
            cumulative_probability += probability
            if cumulative_probability >= random_value:
                return customer_table[customer_id]

        return None # Should not happen

    def construct_solution(self, pheromone_matrix, customer_table, alpha, beta):
        """Constructs a complete solution for the CVRP, considering max distance."""
        self.reset()
        num_served_customers = 0
        while num_served_customers < self.num_customers:
            current_node = self.current_route.customers[-1] if self.current_route.customers else self.depot
            next_customer = self.select_next_customer(pheromone_matrix, customer_table, alpha, beta)

            if next_customer:
                distance = calculate_distance(current_node, next_customer)
                self.current_route.add_customer(next_customer, distance)
                self.current_vehicle.load(next_customer.demand)
                self.visited_customers[next_customer.id - 1] = True
                num_served_customers += 1
            else:
                # No feasible customer can be added to the current route, close the route and start a new one
                if self.current_route.customers:
                    self.current_route.current_distance += calculate_distance(current_node, self.depot) # Return to depot
                self.current_route = Route(self.depot, self.max_distance)
                self.current_route.max_capacity = self.capacity
                self.solution.add_route(self.current_route)
                self.current_vehicle = Vehicle(self.capacity, self.max_distance)

                # Try to find any unvisited customer to start the new route
                unvisited = [cust for i, cust in customer_table.items() if not self.visited_customers[i - 1]]
                if unvisited:
                    # Force visit to an unvisited customer if possible and within max distance for a round trip
                    feasible_starts = []
                    for first_customer in unvisited:
                        round_trip_distance = calculate_distance(self.depot, first_customer) + calculate_distance(first_customer, self.depot)
                        if round_trip_distance <= self.max_distance and self.current_vehicle.can_accommodate(first_customer.demand):
                            feasible_starts.append(first_customer)

                    if feasible_starts:
                        first_customer = random.choice(feasible_starts)
                        distance = calculate_distance(self.depot, first_customer)
                        self.current_route.add_customer(first_customer, distance)
                        self.current_vehicle.load(first_customer.demand)
                        self.visited_customers[first_customer.id - 1] = True
                        num_served_customers += 1
                    else:
                        break # No feasible unvisited customer to start a new route within max distance
                else:
                    break # All customers visited

        # Final return to depot for the last route if it has customers
        if self.current_route.customers:
            self.current_route.current_distance += calculate_distance(self.current_route.customers[-1], self.depot)

        return self.solution

def initialize_pheromone_matrix(depot, customer_table, initial_pheromone=1.0):
    """Initializes the pheromone matrix with a constant value."""
    num_nodes = len(customer_table) + 1 # Including the depot
    pheromone_matrix = [[initial_pheromone] * num_nodes for _ in range(num_nodes)]
    return pheromone_matrix

def update_pheromone(pheromone_matrix, solutions, evaporation_rate):
    """Updates the pheromone matrix based on the solutions found by the ants."""
    # Pheromone evaporation
    for i in range(len(pheromone_matrix)):
        for j in range(len(pheromone_matrix[i])):
            pheromone_matrix[i][j] *= (1 - evaporation_rate)

    # Pheromone deposit
    for solution in solutions:
        cost = solution.calculate_total_distance()
        if cost > 0:
            pheromone_deposit = 1.0 / cost
            for route in solution.routes:
                path = [solution.depot] + route.customers + [solution.depot]
                for i in range(len(path) - 1):
                    node1_id = path[i].id if isinstance(path[i], Customer) else 0
                    node2_id = path[i+1].id if isinstance(path[i+1], Customer) else 0
                    pheromone_matrix[node1_id][node2_id] += pheromone_deposit
                    pheromone_matrix[node2_id][node1_id] += pheromone_deposit # Assuming symmetric distances

def ant_colony_optimization(depot, customer_table, capacity, max_distance, num_ants, alpha, beta, evaporation_rate, num_iterations, optimal_value):
    """Runs the Ant Colony Optimization algorithm for CVRP with max distance constraint."""
    num_customers = len(customer_table)
    pheromone_matrix = initialize_pheromone_matrix(depot, customer_table)
    best_solution = None
    best_distance = float('inf')
    distance_table = []

    iteration = 0
    last_correction = 0
    while ((best_distance > optimal_value) and (iteration < num_iterations)):
        all_ant_solutions = []
        for i in range(num_ants):
            ant = Ant(i, num_customers, capacity, depot, max_distance)
            solution = ant.construct_solution(pheromone_matrix, customer_table, alpha, beta)
            if is_solution_valid(solution, customer_table, capacity, max_distance):
                all_ant_solutions.append(solution)
                total_distance = solution.calculate_total_distance()
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_solution = solution
                    last_correction = iteration

        update_pheromone(pheromone_matrix, all_ant_solutions, evaporation_rate)
        print(f"Iteration {iteration + 1}: Best distance = {best_distance:.2f}")
        distance_table.append(best_distance)
        iteration += 1

    return best_solution, best_distance, distance_table, last_correction

if __name__ == '__main__':
    file_path = '/home/piotr/PycharmProjects/ants/Ants/data/Vrp-Set-E/E/E-n22-k4.vrp'
    number_of_trucks, optimal_value, capacity, depot, customer_table = read_vrp_file(file_path)

    # ACO parameters
    num_ants = 4
    alpha = 0.5
    beta = 7.0
    evaporation_rate = 0.8
    num_iterations = 100

    best_solution, best_distance = ant_colony_optimization(
        depot, customer_table, capacity, num_ants, alpha, beta, evaporation_rate, num_iterations
    )

    print("\n--- Best Solution Found ---")
    if best_solution:
        print(f"Total distance: {best_distance:.2f}")
        for i, route in enumerate(best_solution.routes):
            print(f"Route {i + 1}: {route}")
        print(f"Is solution valid: {is_solution_valid(best_solution, customer_table, capacity)}")
    else:
        print("No valid solution found.")