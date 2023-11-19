import pandas as pd
import matplotlib.pyplot as plt
import os
from random import randint
from KmeansConstrained import KMeansConstrained

from dataset import DataLoader
from util import *

# 1. csv 데이터를 읽어, (위경도 좌표 tuple, area, polygon)로 구성된 객체로 이뤄진 Series로 만든다.abs
"""_summary_
input : csv data
output : Series of Field object

Field's attribute : x,y coordinate, Area
"""


# 2. CSP를 위한 Var, domain 정의
"""_summary_
Var : Field's point
domain : The number of users (Clustering Result Set)
"""

# 3. Constraints 정의
"""_summary_
1. Hard Constraints :
    - 하나의 필지는 반드시 하나의 클러스터링 안에 들어가야 한다.

2. Soft Constraints :
    - 기본은 사용자가 원하는 양만큼의 토지를 할당
    - 별도 요구사항이 없는 경우 토지 균등 분배

"""


if __name__ == '__main__':
    # 메인 함수 돌리기 전에 필요한 Setup
    folder_path = './asset/'
    files = os.listdir(folder_path)

    for file_name in files:
        
        data_loader = DataLoader(folder_path+file_name)
        dataset = data_loader.dataset

        # 좌표 목록과 area 목록
        coordes = data_loader.convert_to_coordes()
        areas = list(dataset['area'])
    
        # random 모듈을 사용하여 n개의 10000 이상의 난수로 구성된 리스트 생성
        num_clusters = 3
        min_area = 10000  
        required_areas = [randint(min_area, min_area + 10000) for _ in range(num_clusters)]
    
        # constrained input : areas, required_areas
    
        kmeans_constrained = KMeansConstrained(areas, required_areas)
        kmeans_constrained.fit_predict(coordes)
        save_visualization_result(dataset['x'],dataset['y'],clf.labels_)