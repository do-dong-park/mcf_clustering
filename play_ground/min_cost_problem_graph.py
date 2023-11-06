import numpy as np
from sklearn.utils.extmath import row_norms, squared_norm, cartesian

def minimum_cost_flow_problem_graph(X, A, C, D, cluster_sizes):
    # Setup minimum cost flow formulation graph
    # Vertices indexes:
    # X-nodes: [0, n(x)-1], 
    # C-nodes: [n(X), n(X)+n(C)-1], 
    # C-dummy nodes:[n(X)+n(C), n(X)+2*n(C)-1],
    # Artificial node: [n(X)+2*n(C), n(X)+2*n(C)+1-1]

    # Create indices of nodes
    # 필지 노드 수
    n_X = len(X)
    # 클러스터 중심점 노드 수
    n_C = C.shape[0]
    
    # 필지 노드의 인덱스
    X_ix = np.arange(n_X)
    # 클러스터링용 더미 중심 노드
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    # 면적 고려 용 중심점 노드 인덱스
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    # 인공 노드 인덱스
    art_ix = C_ix[-1] + 1
    print("==================={node index}=======================")
    print(f"X_index : {X_ix}")
    print(f"C_index : {C_ix}")
    print(f"art_index : {art_ix}")

    # Edges
    edges_X_C_dummy = cartesian(
        [X_ix, C_dummy_ix]
    )  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack(
        [C_dummy_ix, C_ix], axis=1
    )  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack(
        [C_ix, art_ix * np.ones(n_C)], axis=1
    )  # All C connect to artificial node

    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])
    print("==================={edges}=======================")
    print(f"edges_X_C_dummy : {edges_X_C_dummy}")
    print(f"edges_C_dummy_C : {edges_C_dummy_C}")
    print(f"edges_C_arts : {edges_C_art}")

    # Costs
    # 필지부터 더미 중심점 간 비용 (거리 및 면적 고려)
    # 필지 수 * 중심점 수
    costs_X_C_dummy = D.reshape(D.size)
    costs_C_dummy_C = np.zeros(len(edges_C_dummy_C))
    costs_C_art = np.zeros(len(edges_C_art))
    costs = np.concatenate([costs_X_C_dummy,costs_C_dummy_C,costs_C_art])
    
    print("==================={edges}=======================")
    print(f"costs_X_C_dummy : {costs_X_C_dummy}")
    print(f"costs_C_dummy_C : {costs_C_dummy_C}")
    print(f"costs_C_arts : {costs_C_art}")

    # Capacities - can set for max-k
    # 중심 노드와 더미 중심 노드 간 용량
    
    capacities_X_C_dummy = np.ones(edges_X_C_dummy.shape[0])
    cap_non = n_X # The total supply and therefore wont restrict flow
    capacities_C_dummy_C = cap_non * np.ones(n_C)
    
    tolerance = 0.0
    adjusted_cluster_sizes = np.array(cluster_sizes) * (1 + tolerance)
    capacities_C_art = adjusted_cluster_sizes
    
    capacities = np.concatenate([capacities_X_C_dummy, capacities_C_dummy_C,capacities_C_art])
    
    print("==================={capacity}=======================")
    print(f"capacities_X_C_dummy : {capacities_X_C_dummy}")
    print(f"capacities_C_dummy_C : {capacities_C_dummy_C}")
    print(f"capacities_C_arts : {capacities_C_art}")

    supplies_X = np.array(A)
    demands_C_dummy = np.zeros(n_C)
    demands_C = -1 * np.array(cluster_sizes)
    demands_art = -1 * (np.sum(supplies_X)+np.sum(demands_C))
    
    supplies = np.concatenate([supplies_X, demands_C_dummy, demands_C, [demands_art]])
    print("==================={Supplies & Demands}=======================")
    print(f"supplies X : {A}")
    print(f"demands_C_dummy : {demands_C_dummy}")
    print(f"demands_C : {demands_C}")
    print(f"demands_art : {demands_art}")
    print(f"supplies : {supplies}")

    edges = edges.astype("int32")
    costs = np.around(costs * 1000, 0).astype("int32")
    capacities = capacities.astype("int32")
    supplies = supplies.astype("int32")
    
    print("==================={Results}=======================")
    print(f"edges size : {edges.shape[0]}")
    print(f"costs : {len(costs)}")
    print(f"capacities : {len(capacities)}")
    print(f"supplies : {len(supplies)}")

    return edges, costs, capacities, supplies, n_C, n_X
