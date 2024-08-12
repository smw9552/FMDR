import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from algorithm.models import models
import random
import tensorflow as tf

m = models()


#set seed
random_seed = 2024
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)
tf.random.set_seed(random_seed)


# load file info
input_file_obs = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\mixture_drc\\20240809_mixture_dataset.xlsx"
input_file_dir = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\single_chemical_drc\\"
output_file_path_image = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\result\\mdr_graph\\"
output_file_path_mdr = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\result\\"
output_file_path_fmdr = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\result\\"


# KRICT Server
module_path = "C:\\Users\\User\\PycharmProjects\\FMDR\\algorithm\\"
rm_module_name = 'rm_python_opt'
exec(open(module_path + str("rm_python_opt.py"), "r", encoding='utf-8').read())


'''
with open("C:\\Users\\user\\PycharmProjects\\FMDR\\algorithm\\rm_python_opt.py", encoding='utf-8') as f:
    exec(f.read())
'''


#batch file list
dir_filename = os.listdir(input_file_dir)
files = [file for file in dir_filename if file.endswith(".csv")]
file_list = []

for ai in range(0, len(files)):
    file_list.append(input_file_dir + files[ai])

#파일 리스트 번호를 순서대로 정렬하기 위해 함수 추가
def sort_files(file_list):
    # 파일 번호를 추출하여 정수로 변환하는 함수
    def extract_number(filename):
        # 정규표현식을 사용하여 파일 번호 추출
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return float('inf')  # 숫자가 없는 경우 매우 큰 수로 처리 (이 경우에는 파일 이름을 뒤로 보내게 됨)

    # 파일 리스트를 파일 번호를 기준으로 정렬
    sorted_files = sorted(file_list, key=extract_number)

    return sorted_files

sorted_files = sort_files(file_list)
print(sorted_files)

sorted_file_name = []
for file in sorted_files:
    temp_file = str(file).replace(str(input_file_dir), "")
    sorted_file_name.append(temp_file.replace(".csv", ""))


######################
#Set range of effects#
######################
effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#effect = np.linspace(0.1, 0.8, 1000)


################
## Experiment ##
################

ecPoints_obs, name_list = m.exp_model(input_file_obs, module_path, rm_module_name, effect)

result_final_df_obs = pd.DataFrame(ecPoints_obs, columns=effect, index=name_list)
final_df_obs = result_final_df_obs.transpose()


################
## Prediction ##
################

ecPoints_ca = []
for bi in range(0, len(sorted_files)):
    ca = m.ca_model(sorted_files[bi], module_path, rm_module_name, random_seed, effect)
    ecPoints_ca.append(ca)

result_df_pred = pd.DataFrame(ecPoints_ca, columns=effect, index=name_list)
final_df_pred = result_df_pred.transpose()



####################
##FMDR calculation##
####################

## (Mean MDR) 전 구간에 대한 평가 ##

full_curve_mdr_list = []

for name in name_list:
    conc_obs = final_df_obs[name].tolist()
    conc_pred = final_df_pred[name].tolist()
    #effect = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    #obs 데이터 기준 결측치(nan) 검토
    if np.isnan(conc_obs).any():
        nan_indices_obs = np.where(np.isnan(conc_obs))[0]
        conc_obs_F1 = [value for i, value in enumerate(conc_obs) if i not in nan_indices_obs]  # obs에서 검토
        conc_pred_F1 = [value for i, value in enumerate(conc_pred) if i not in nan_indices_obs]  # obs에서 통일시키기 위해서 정리
        effect_F1 =  [value for i, value in enumerate(effect) if i not in nan_indices_obs]  # obs에서 통일시키기 위해서 정리

    else:
        conc_obs_F1 = conc_obs
        conc_pred_F1 = conc_pred
        effect_F1 = effect


    #pred 데이터 기준 결측치(nan) 검토
    if np.isnan(conc_pred).any():
        nan_indices_pred = np.where(np.isnan(conc_pred))[0]
        conc_obs_F2 = [value for i, value in enumerate(conc_obs_F1) if i not in nan_indices_pred]  # obs에서 검토
        conc_pred_F2 = [value for i, value in enumerate(conc_pred_F1) if i not in nan_indices_pred]  # obs에서 통일시키기 위해서 정리
        effect_F2 = [value for i, value in enumerate(effect_F1) if i not in nan_indices_pred]  # obs에서 통일시키기 위해서 정리

    else:
        conc_obs_F2 = conc_obs_F1
        conc_pred_F2 = conc_pred_F1
        effect_F2 = effect_F1


    #결측치 데이터 제외한 나머지로 MDR 계산 수행
    temp_mdr_list = []
    for di in range(0, len(conc_obs_F2)): #conc_pred_F2와 사이즈 동일함
        temp_mdr_list.append(conc_pred_F2[di] / conc_obs_F2[di])

    full_curve_mdr_list.append(np.mean(temp_mdr_list)) #평균 mdr 계산 결과를 추가

    '''
    ## 그래프 그리고 저장
    plt.plot(conc_obs_F2, effect_F2, '-*', label="Obs", color='black')
    plt.plot(conc_pred_F2, effect_F2, '-*', label="Pred (CA model)", color='blue')


    #plt.xscale('log')
    plt.title(name)
    plt.xlabel('Concentration')
    plt.ylabel('Effect(%)')
    plt.legend()
    #plt.show()


    output_file = output_file_path_image + str(name) + "_result.jpg"
    plt.savefig(output_file)
    plt.clf()  # 그래프 초기화 (초기화하지 않으면 누적됨)
    '''



result_df_mdr = pd.DataFrame(full_curve_mdr_list, index=name_list)


fmdr_df = pd.DataFrame({
    'name': name_list,
    'Full curve MDR': full_curve_mdr_list

})

output_file_fmdr = output_file_path_fmdr + "fmdr_result_normal_points.xlsx"
fmdr_df.to_excel(output_file_fmdr, index=False)




#####################
##Point calculation##
#####################


## (Point MDR) EC10, 50 기준 평가 ##
point_mdr_10_list = []
point_mdr_30_list = []
point_mdr_50_list = []
point_mdr_70_list = []

for name in name_list:
    conc_obs = final_df_obs[name].tolist()
    conc_pred = final_df_pred[name].tolist()

    #obs 데이터 기준 결측치(nan) 검토
    if np.isnan(conc_obs).any():
        nan_indices_obs = np.where(np.isnan(conc_obs))[0]
        conc_obs_F1 = [value for i, value in enumerate(conc_obs) if i not in nan_indices_obs]  # obs에서 검토
        conc_pred_F1 = [value for i, value in enumerate(conc_pred) if i not in nan_indices_obs]  # obs에서 통일시키기 위해서 정리

    else:
        conc_obs_F1 = conc_obs
        conc_pred_F1 = conc_pred

    #pred 데이터 기준 결측치(nan) 검토
    if np.isnan(conc_pred).any():
        nan_indices_pred = np.where(np.isnan(conc_pred))[0]
        conc_obs_F2 = [value for i, value in enumerate(conc_obs_F1) if i not in nan_indices_pred]  # obs에서 검토
        conc_pred_F2 = [value for i, value in enumerate(conc_pred_F1) if i not in nan_indices_pred]  # obs에서 통일시키기 위해서 정리

    else:
        conc_obs_F2 = conc_obs_F1
        conc_pred_F2 = conc_pred_F1


    point_mdr_10_list.append(conc_pred_F2[0] / conc_obs_F2[0])
    point_mdr_30_list.append(conc_pred_F2[2] / conc_obs_F2[2])
    point_mdr_50_list.append(conc_pred_F2[4] / conc_obs_F2[4])
    point_mdr_70_list.append(conc_pred_F2[6] / conc_obs_F2[6])


## MDR 비교 데이터 표 생산 및 저장 ##
mdr_df = pd.DataFrame({
    'name': name_list,
    'Point MDR (EC10)': point_mdr_10_list,
    'Point MDR (EC30)': point_mdr_30_list,
    'Point MDR (EC50)': point_mdr_50_list,
    'Point MDR (EC70)': point_mdr_70_list,
    'Full curve MDR': full_curve_mdr_list

})

output_file_mdr = output_file_path_mdr + "point_mdr_result.xlsx"

mdr_df.to_excel(output_file_mdr, index=False)









