import numpy as np
from sklearn.utils.extmath import row_norms, squared_norm, cartesian

def minimum_cost_flow_problem_graph(X, A, C, D, R):
    # Vertices indexes:
    # X-nodes: [0, n(x)-1], (필지 노드)
    # C-count nodes: [n(X), n(X)+n(C)-1], (클러스터링 노드)
    # C-area nodes:[n(X)+n(C), n(X)+2*n(C)-1], (클러스터 별 요구 면적 노드)
    # Artificial node: [n(X)+2*n(C), n(X)+2*n(C)+1-1] (수요 공급 균형 노드)

    # Create indices of nodes
    # 필지 노드 수
    n_X = len(X)
    # 클러스터 중심점 노드 수
    n_C = C.shape[0]
    
    # 필지 노드의 인덱스
    X_ix = np.arange(n_X)
    # 클러스터링 노드
    C_count_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    # 클러스터 별 요구 면적 노드
    C_area_ix = np.arange(C_count_ix[-1] + 1, C_count_ix[-1] + 1 + n_C)
    # 인공 노드 인덱스
    art_ix = C_area_ix[-1] + 1
    
    print("==================={node index}=======================")
    print(f"X_index : {X_ix}")
    print(f"C_count_index : {C_count_ix}")
    print(f"C_area_index : {C_area_ix}")
    print(f"art_index : {art_ix}")

    # Edges
    edges_X_C_count = cartesian(
        [X_ix, C_count_ix]
    )  # All X's connect to all C count nodes 
    edges_C_count_C_area = np.stack(
        [C_count_ix, C_area_ix], axis=1
    )  # Each C count connects to a corresponding required C area
    edges_C_area_art = np.stack(
        [C_area_ix, art_ix * np.ones(n_C)], axis=1
    )  # All C area node connect to artificial node

    edges = np.concatenate([edges_X_C_count, edges_C_count_C_area, edges_C_area_art])
    print("==================={edges}=======================")
    print(f"edges_X_C_count : {edges_X_C_count}")
    print(f"edges_C_count_C_area : {edges_C_count_C_area}")
    print(f"edges_C_arts : {edges_C_area_art}")

    # Costs
    # 필지에서부터 중심점까지 거리
    costs_X_C_count = D.reshape(D.size)
    # 클러스터와 요구 면적 간 비용 (클러스터의 면적과 요구 면적을 계산하는 목적이므로, 비용은 없음)
    # 클러스터 면적 합 - 요구 면적으로 설정하고 싶은데 그래프 정의 단계에서는 지원하지 않음 (클러스터가 결정 되고나서 면적이 정해지므로)
    costs_C_count_C_area = np.zeros(len(edges_C_count_C_area))
    # 요구 면적과 균형 노드 간 비용
    costs_C_area_art = np.zeros(len(edges_C_area_art))
    # 간선에 대한 각 비용
    costs = np.concatenate([costs_X_C_count,costs_C_count_C_area,costs_C_area_art])
    
    print("==================={edges}=======================")
    print(f"costs_X_C_count : {costs_X_C_count}")
    print(f"costs_C_count_C_area : {costs_C_count_C_area}")
    print(f"costs_C_arts : {costs_C_area_art}")

    # Capacities
    
    # 필지는 클러스터에 한 개씩 할당되어야 함
    capacities_X_C_count = np.ones(edges_X_C_count.shape[0])
    
    required_area = R
    # 클러스터의 면적은 요구 면적과 비슷해야 함
    tolerance = 1.5
    capacities_C_count_C_area = required_area * np.ones(n_C) * tolerance
    
    # 면적 노드와 균형 노드 간 제약은 없음
    capacities_non = np.sum(R)
    capacities_C_area_art = np.ones(n_C) * np.sum(R)
    
    # 수용량 각 간선에 부여
    capacities = np.concatenate([capacities_X_C_count, capacities_C_count_C_area,capacities_C_area_art])
    
    print("==================={capacity}=======================")
    print(f"capacities_X_C_count : {capacities_X_C_count}")
    print(f"capacities_C_count_C_area : {capacities_C_count_C_area}")
    print(f"capacities_C_arts : {capacities_C_area_art}")

    # 필지 노드 (갯수 도메인)
    supplies_X = np.ones(n_X)
    # 클러스터 별 면적 공급 노드 (면적 도메인)
    # 문제점 : 클러스터 이후, 면적값을 고려 해야 하는데 엔진 구조적으로 불가능
    supplies_for_C_area = np.zeros(n_C)
    # 면적에 대한 수요 노드
    demands_C_area = -1 * np.array(R)
    # 균형 노드는 면적에 대한 합
    demands_art = [-1 * np.sum(A)]
    
    supplies = np.concatenate([supplies_X, supplies_for_C_area, demands_C_area, demands_art])
    print("==================={Supplies & Demands}=======================")
    print(f"supplies X : {supplies_X}")
    print(f"supplies_for_C_area : {supplies_for_C_area}")
    print(f"demands_C_area : {demands_C_area}")
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
    print(f"supplies & Demandes : {len(supplies)}")

    return edges, costs, capacities, supplies, n_C, n_X
