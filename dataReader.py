import re

class Depot:
    """Represents the depot with coordinates."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Depot"

class Customer:
    """Represents a customer with demand and coordinates."""
    def __init__(self, id, x, y, demand):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand

    def __repr__(self):
        return f"Customer(id={self.id}, demand={self.demand})"

def read_vrp_file(file_path):
    """
    Reads a .vrp file and extracts relevant information for CVRP.
    Now assumes customer IDs start from 0 (depot is 0).

    Args:
        file_path (str): The path to the .vrp file.

    Returns:
        tuple: A tuple containing:
            - number_of_trucks (int): The minimum number of trucks (K).
            - optimal_value (int): The optimal value (if specified in COMMENT).
            - capacity (int): The capacity of each truck (Q).
            - depot (Depot): A Depot object representing the depot location (node 0).
            - customer_table (dict): A dictionary where keys are customer IDs
              (starting from 1) and values are Customer objects.
    """
    number_of_trucks = None
    optimal_value = None
    capacity = None
    node_coords = {}
    demands = {}
    customer_table = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    reading_nodes = False
    reading_demands = False
    dimension = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("NAME"):
            pass  # We don't need the name for now
        elif line.startswith("COMMENT"):
            match_trucks = re.search(r"Min no of trucks: (\d+)", line)
            if match_trucks:
                number_of_trucks = int(match_trucks.group(1))
            match_optimal = re.search(r"Optimal value: (\d+)", line)
            if match_optimal:
                optimal_value = int(match_optimal.group(1))
        elif line.startswith("TYPE") and "CVRP" not in line:
            raise ValueError("File is not a CVRP type")
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE") and "EUC_2D" not in line:
            raise ValueError("Edge weight type is not EUC_2D. This reader only supports EUC_2D.")
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            reading_nodes = True
            continue
        elif line.startswith("DEMAND_SECTION"):
            reading_nodes = False
            reading_demands = True
            continue
        elif line.startswith("DEPOT_SECTION"):
            reading_demands = False
            continue
        elif line.startswith("EOF"):
            continue

        if reading_nodes:
            parts = line.split()
            node_id = int(parts[0]) - 1  # Change to 0-based indexing
            x = int(parts[1])
            y = int(parts[2])
            node_coords[node_id] = (x, y)
        elif reading_demands:
            parts = line.split()
            customer_id = int(parts[0]) - 1  # Change to 0-based indexing
            demand = int(parts[1])
            demands[customer_id] = demand

    if 0 not in node_coords:
        raise ValueError("Depot (node 0) coordinates not found.")
    if 0 not in demands or demands[0] != 0:
        raise ValueError("Depot (node 0) demand should be 0.")

    depot_x, depot_y = node_coords[0]
    depot = Depot(depot_x, depot_y)

    for customer_id in range(1, dimension): # Iterate from 1 to dimension-1 for customers (0 is depot)
        if customer_id not in node_coords or customer_id not in demands:
            raise ValueError(f"Missing coordinates or demand for customer {customer_id + 1} (now id {customer_id})")
        x, y = node_coords[customer_id]
        demand = demands[customer_id]
        customer_table[customer_id] = Customer(customer_id, x, y, demand)

    return number_of_trucks, optimal_value, capacity, depot, customer_table