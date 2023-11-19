"""Linear assignment example."""
import numpy as np
from ortools.graph.python import min_cost_flow


def mcf_solver(areas: np.ndarray, requested_areas: np.ndarray, costs: np.ndarray, tolerence=0):
    """
    Solving an Assignment Problem with MinCostFlow.
    area: 필지 면적 리스트
    requested_areas: 팀들의 요청된 면적
    costs: 중심점으로부터 필지간의 거리 행렬
    """
    print(f"areas : {np.sum(areas)}")
    print(f"requested_area : {np.sum(requested_areas)}")
    print(f"costs : {costs}")
    
    requested_areas = np.array (requested_areas) +tolerence
    num_areas = len(areas)
    num_teams = len(requested_areas)
    total_area = np.sum(areas)
    
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()
    
    source = 0
    area_arr = np.arange(1, num_areas + 1)
    team_arr = np.arange(num_areas + 1, num_teams + num_areas + 1)
    sink = num_teams + num_areas + 1
    
    # Define the directed graph for the flow.
    start_nodes = np.concatenate(
        (np.repeat(source, num_areas),
        np.repeat(area_arr, num_teams),
        team_arr)
    )
    
    end_nodes = np.concatenate(
        (area_arr,
        np.repeat([team_arr], num_areas, axis=0).reshape(-1),
        np.repeat(sink, num_teams))
    )
    
    capacities = np.concatenate(
        (areas,
        np.repeat(areas, num_teams),
        requested_areas)
    )
    
    costs = np.concatenate(
        ([0] * num_areas,
        costs.reshape(-1),
        [0] * num_teams)
    )
    
    supplies = np.zeros(sink + 1, dtype=int)
    supplies[0] = total_area
    supplies[-1] = -total_area
    
    # Add each arc.
    smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, costs
    )
    
    # Add node supplies.
    smcf.set_nodes_supplies(np.arange(sink + 1), supplies)

    # Find the minimum cost flow between node 0 and node 10.
    status = smcf.solve()

    if status == smcf.OPTIMAL:
        print("Total cost = ", smcf.optimal_cost())
        print()
        for arc in range(smcf.num_arcs()):
            # Can ignore arcs leading out of source or into sink.
            if smcf.tail(arc) != source and smcf.head(arc) != sink:
                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.
                if smcf.flow(arc) > 0:
                    pass
                    # print(
                    #     "Worker %d assigned to task %d.  Cost = %d"
                    #     % (smcf.tail(arc), smcf.head(arc), smcf.unit_cost(arc))
                    # )
    else:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        
    # Assignment
    labels_M = (
        np.array([smcf.flow(i) for i in range(len(areas) * len(requested_areas))])
        .reshape(len(areas), len(requested_areas))
        .astype("int32")
    )

    labels = labels_M.argmax(axis=1)
    
    return labels