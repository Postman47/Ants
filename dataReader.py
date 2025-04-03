import itertools
import os

def parse_cvrp_file(filepath):
    """
    Parses a CVRP file and extracts relevant data.

    Args:
        filepath (str): The path to the CVRP file.

    Returns:
        dict: A dictionary containing extracted data (dimension, capacity, coordinates, demands, depot).
              Returns None if an error occurs.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        data = {}
        data['coordinates'] = {}
        data['demands'] = {}
        data['depot'] = None

        coord_section = False
        demand_section = False
        depot_section = False

        for line in lines:
            line = line.strip()

            if "DIMENSION" in line:
                data["dimension"] = int(line.split(":")[1].strip())
            elif "CAPACITY" in line:
                data["capacity"] = int(line.split(":")[1].strip())
            elif "NODE_COORD_SECTION" in line:
                coord_section = True
                demand_section = False
                depot_section = False
                continue
            elif "DEMAND_SECTION" in line:
                coord_section = False
                demand_section = True
                depot_section = False
                continue
            elif "DEPOT_SECTION" in line:
                coord_section = False
                demand_section = False
                depot_section = True
                continue
            elif "EOF" in line:
                break

            if coord_section:
                parts = line.split()
                if len(parts) == 3:
                    node_id, x, y = map(float, parts)
                    data['coordinates'][int(node_id)] = (x, y)
            elif demand_section:
                parts = line.split()
                if len(parts) == 2:
                    node_id, demand = map(int, parts)
                    data['demands'][int(node_id)] = demand
            elif depot_section:
                if line.isdigit():
                    data['depot'] = int(line)

        return data

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def create_instance_config(filepath):
    """
    Creates an instance config from the provided filepath.
    Args:
        filepath: path to the .vrp instance file.
    Returns:
        dict: config of the instance
    """
    data = parse_cvrp_file(filepath)
    if data:
        nodes = list(data["coordinates"].keys())
        depot_id = data["depot"]
        customers = [node for node in nodes if node != depot_id]
        return {
            "depot": data["coordinates"][depot_id],
            "customers": [(customer, data["coordinates"][customer], data["demands"][customer]) for customer in customers],
            "num_vehicles": None, # needs to be added later if known
            "vehicle_capacity": data["capacity"],
            "max_distance": float('inf'), # or some other constraint, add later if needed.
        }
    return None

def generate_experiment_configs(directory):
    """
    Generates experiment configurations for CVRP files in a directory.

    Args:
        directory (str): The path to the directory containing CVRP files.

    Returns:
        list: A list of dictionaries, where each dictionary represents an experiment configuration.
    """

    configs = []

    # Define parameter ranges for ACO algorithms
    num_ants_values = [10, 20, 30]
    max_iterations_values = [50, 100, 150]
    alpha_values = [1, 2]
    beta_values = [2, 3]
    rho_values = [0.1, 0.5]
    q_values = [100]
    q0_values = [0.9]
    tau_0_values = [0.1]

    for filename in os.listdir(directory):
        if filename.endswith(".vrp"):
            filepath = os.path.join(directory, filename)
            instance_config = create_instance_config(filepath)
            if instance_config:
                # Basic ACO configs
                for num_ants, max_iterations, alpha, beta, rho, q in itertools.product(
                        num_ants_values, max_iterations_values, alpha_values, beta_values, rho_values, q_values):
                    configs.append({
                        "filename": filename,
                        "algorithm": "basic_aco",
                        "params": {
                            "num_ants": num_ants,
                            "max_iterations": max_iterations,
                            "alpha": alpha,
                            "beta": beta,
                            "rho": rho,
                            "q": q,
                        },
                        "instance_config": instance_config
                    })

                # Ant System configs
                for num_ants, max_iterations, alpha, beta, rho, q in itertools.product(
                        num_ants_values, max_iterations_values, alpha_values, beta_values, rho_values, q_values):
                    configs.append({
                        "filename": filename,
                        "algorithm": "ant_system",
                        "params": {
                            "num_ants": num_ants,
                            "max_iterations": max_iterations,
                            "alpha": alpha,
                            "beta": beta,
                            "rho": rho,
                            "q": q,
                        },
                        "instance_config": instance_config
                    })

                # Ant Colony System configs
                for num_ants, max_iterations, alpha, beta, rho, q, q0, tau_0 in itertools.product(
                        num_ants_values, max_iterations_values, alpha_values, beta_values, rho_values, q_values, q0_values, tau_0_values):
                    configs.append({
                        "filename": filename,
                        "algorithm": "ant_colony_system",
                        "params": {
                            "num_ants": num_ants,
                            "max_iterations": max_iterations,
                            "alpha": alpha,
                            "beta": beta,
                            "rho": rho,
                            "q": q,
                            "q0": q0,
                            "tau_0": tau_0,
                        },
                        "instance_config": instance_config
                    })

                # Greedy config (no parameters)
                configs.append({
                    "filename": filename,
                    "algorithm": "greedy",
                    "params": {},
                    "instance_config": instance_config
                })
    return configs