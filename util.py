import time
import datetime
import logging
import random
import json
import pandas as pd
import matplotlib.pyplot as plt

def time_measurement(f):
    def wrap(*args):
        start_r = time.perf_counter()
        start_p = time.process_time()
        # 함수 실행
        ret = f(*args)
        end_r = time.perf_counter()
        end_p = time.process_time()
        elapsed_r = end_r - start_r
        elapsed_p = end_p - start_p

        print(f'{f.__name__} elapsed: {elapsed_r:.6f}sec (real) / {elapsed_p:.6f}sec (cpu)')
        return ret
   # 함수 객체를 return
    return wrap

def get_now_datetime():
    # 현재 시각 가져오기
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y.%m.%d %H:%M")
    return timestamp

def save_checkpoint_clustering_results(input_filename, output_filename, cluster_id):
    '''
    클러스터링 결과 체크 포인트를 저장하는 함수 (토지 데이터 + 군집화 결과 데이터)
    :param output_filename: 결과 파일 이름
    :param input_filename: 읽어올 토지 정보 파일 이름
    :param cluster_id: 클러스터링 결과 id
    :param cluster_area: 클러스터링 결과 면적값
    :return: checkpoint (csv)
    '''
    folder_path = "./data/"
    lot_df = pd.read_csv(folder_path+input_filename, low_memory=False, encoding='UTF-8')

    cluster_id_df = pd.DataFrame(cluster_id, columns=['cluster_id'])
    # 클러스터링 아이디 데이터 타입 Int로 변경
    cluster_id_df = cluster_id_df.astype({"cluster_id": "int"})

    # 필지 데이터와 클러스터링 결과값 병합
    lot_df["cluster_id"] = cluster_id_df["cluster_id"]

    # Excel 파일로 내보내기
    output_filename = f"checkpoint_{output_filename}.csv"
    lot_df.to_csv(f"./checkpoints/{output_filename}_{input_filename}", index=False,encoding='UTF-8')

    print("Checkpoint Save Completed")

    return lot_df


def save_json_result(input_filename,output_filename,checkpoint):
    '''
    check point의 정보를 json으로 반환하는 코드
    :param checkpoint: 결과값
    :return: 반환값 없음 저장만 하면 됨
    '''

    cluster_unique_id_list = checkpoint["cluster_id"].unique().tolist()
    result_dict = {'Algorithm': 'MCF Clustering', 'TaskList': {}}

    cluster_unique_id_result = checkpoint["cluster_id"].unique().tolist()

    for key in cluster_unique_id_result:
        polygon_list_dict = {'PolygonList': {}}
        lot_polygon_list_per_task = checkpoint.loc[
            checkpoint["cluster_id"] == key,checkpoint.columns[4]].tolist()

        for i, polygon in enumerate(lot_polygon_list_per_task):
            polygon_coordinate_per_lot_list = eval(polygon)["coordinates"][0][0]
            polygon_list_dict['PolygonList']["Polygon" + str(i)] = polygon_coordinate_per_lot_list

        result_dict['TaskList']['Task' + str(key)] = polygon_list_dict

    file_path = f"./results/result_{output_filename}_{input_filename}.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(result_dict, file)

    print("Result Json Save Completed")

    return


def save_visualization_result(tmp_x, tmp_y, cluster_id,filename):
    plt.scatter(tmp_x, tmp_y, c=cluster_id)
    plt.savefig(f'./clustering_result_img/{get_now_datetime() + filename}.jpg', dpi=300)

    print("Visualization Save Completed")

    return

def save_visualization_pre_result(timp_x,tmp_y,filename):
    plt.scatter(timp_x,tmp_y)
    plt.savefig(f'./preclustering_img/{get_now_datetime()+ filename}.jpg', dpi=300)
    
def split_integer_randomly(num, n):
    parts = sorted(random.sample(range(1, num), n - 1))
    parts = [0] + parts + [num]
    parts = [parts[i + 1] - parts[i] for i in range(n)]
    return parts