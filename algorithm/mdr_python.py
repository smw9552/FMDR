import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


# load file info
input_file_obs = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\mixture_drc\\20240809_mixture_dataset.xlsx"
input_file_path_pred = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\single_chemical_drc\\"
output_file_path_image = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\result\\mdr_graph\\"
output_file_path_mdr = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\result\\"

with open("C:\\Users\\user\\PycharmProjects\\FMDR\\algorithm\\rm_python_opt.py", encoding='utf-8') as f:
    exec(f.read())


#####################
##experimental data##
#####################


#exp 실험데이터 처리(실험데이터에 대한 DRC 정보를 하나로 정리해둔 경우에만 활용 가능)
#exp 실험데이터 처리를 통한 0~100 사이 구간에서 농도값 계산
df_obs = pd.read_excel(input_file_obs)

name_list = df_obs['name']
model_list = df_obs['RM']
param_a_list = df_obs['a']
param_b_list = df_obs['b']
param_c_list = df_obs['g']

ecPoints_obs = [] #Concentration
effect = [10, 20, 30, 40, 50, 60, 70, 80, 90] #Effect(독성팀 in vitro 데이터는 percentage scale)


for ai in range(0, df_obs.shape[0]):
    print(model_list[ai])
    rm_func = globals().get(model_list[ai]) #함수명으로 전역 네임스페이스에서 검색하여 객체 가져옴
    print(rm_func)

    temp_ecPoints = []
    for bi in range(0, len(effect)):
        temp_ecPoints.append(rm_func(effect[bi], param_a_list[ai], param_b_list[ai], param_c_list[ai], mode='ec'))


    ecPoints_obs.append(temp_ecPoints)

result_df_obs = pd.DataFrame(ecPoints_obs, columns=effect, index=name_list)
final_df_obs = result_df_obs.transpose()




###################
##prediction data##
###################


#pred 데이터 처리(농도를 기준으로 하기 때문에 보수적인 CA 예측모델을 활용)
pred_dir_filename = os.listdir(input_file_path_pred)
files = [file for file in pred_dir_filename if file.endswith(".csv")]
file_list = []
for bi in range(0, len(files)):
    file_list.append(input_file_path_pred + files[bi])

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




## CA 모델 ##
ecPoints_pred = []


for f in range(0, len(sorted_files)):

    print(sorted_files[f])

    df_pred = pd.read_csv(sorted_files[f])

    chem_name_list = df_pred['Substance']
    cas_list = df_pred['CAS NO']
    comp_list = df_pred['percentage']
    model_list = df_pred['RM']
    input_unit_list = df_pred['Unit for RM']
    output_unit_list = df_pred['Coversion Unit']  # Coversion unit을 Conversion Unit으로 수정해야함
    param_a_list = df_pred['a']
    param_b_list = df_pred['b']
    param_c_list = df_pred['g']
    ec50_list = df_pred['EC50']

    #print(model_list)

    # scale check (fraction or percentage)
    # 단일물질 기준 DRC 정보를 활용하기 때문에 scale에 따라 모두 적용 필요
    ec_check_list = []
    for i in range(0, df_pred.shape[0]):
        #print(model_list[i])
        rm_func = globals().get(model_list[i])  # 함수명으로 전역 네임스페이스에서 검색하여 객체를 가져옴
        #print(rm_func)

        if str(param_c_list[i]) != 'nan':  # 파라미터가 3개인 경우
            ec_check_list.append(
                rm_func(ec50_list[i], param_a_list[i], param_b_list[i], param_c_list[i], mode='ce'))
        else:  # 파라미터가 2개인 경우
            ec_check_list.append(rm_func(ec50_list[i], param_a_list[i], param_b_list[i], mode='ce'))

    if sum(ec_check_list) > df_pred.shape[0]:
        # percentage scale (in vitro 실험데이터는 percentage scale에 가까움)
        print("##Percentage scale##")
        scale = "Percentage"
    else:
        # fraction scale
        print("##Fraction scale##")
        scale = "Fraction"

    # composition scale check
    total_comp = sum(comp_list)
    if total_comp <= 1:  # Fraction scale로 입력된 경우(0~1)
        new_comp_list = comp_list
    else:  # Percentage scale로 입력된 경우(0~100)
        new_comp_list = []
        for j in range(0, len(comp_list)):
            new_comp_list.append(comp_list[j] / total_comp)

    # 각 물질별 함수를 이용한 effect에 대한 농도값 계산
    # parameters: alpha(height), beta(slope), gamma(center point)
    # y축 scale 조정을 위해서는 DRC parameter 중 alpha(height)를 조정해야함
    # (중요) (1) Percentage -> Fraction: alpha / 100, (2) Fraction -> Percentage: alpha * 100
    ecPoints = []
    # scale을 fraction으로 고정하므로 effPoint도 fraction에 맞게 고정
    effPoints = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    for ai in range(0, df_pred.shape[0]):
        rm_func = globals().get(model_list[ai])  # 함수명으로 전역 네임스페이스에서 검색하여 객체를 가져옴
        #print(rm_func)
        temp_ecPoints = []

        if scale == "Fraction":
            for bi in range(0, len(effPoints)):
                if str(param_c_list[ai]) != 'nan':
                    temp_ecPoints.append(
                        rm_func(effPoints[bi], param_a_list[ai], param_b_list[ai], param_c_list[ai], mode='ec'))
                else:
                    temp_ecPoints.append(rm_func(effPoints[bi], param_a_list[ai], param_b_list[ai], mode='ec'))
            # rm_func(y, alpha, beta, gamma) - 파라미터 3개 이상인 경우 따로 만들어야 함(추가 제어문 설정 필요)

        else:  # percentage scale
            for bi in range(0, len(effPoints)):
                if str(param_c_list[ai]) != 'nan':
                    temp_ecPoints.append(
                        rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], param_c_list[ai], mode='ec'))
                else:
                    temp_ecPoints.append(
                        rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], mode='ec'))
            # rm_func(y, alpha, beta, gamma) - 파라미터 3개 이상인 경우 따로 만들어야 함(추가 제어문 설정 필요)

        ecPoints.append(temp_ecPoints)

    # Effect 결과를 계산하기 위해 CA 계산을 통해 농도구간 산출
    cal_ecPoints = []
    # 개별물질 함량 값을 effPoints의 농도값으로 나누는 과정
    for ci in range(0, df_pred.shape[0]):
        cal_ecPoints.append([(new_comp_list[ci] / element) for element in ecPoints[ci]])

    # 개별물질별로 나누어진 값을 합산하는 과정
    sum_cal_ecPoints = [sum(sublist) for sublist in zip(*cal_ecPoints)]

    # 합산된 결과에 최종 역수를 취함 (scale에 따라 결과 정리 필요)
    # fraction scale 기준으로 출력
    ca_result = [1 / element for element in sum_cal_ecPoints]

    ## CA 결과 기준으로 nan 체크 ##
    if np.isnan(ca_result).any():  # nan이 포함된 경우
        # nan 위치 확인 및 nan 제거
        nan_indices = np.where(np.isnan(ca_result))[0]
        final_ca_result = [value for i, value in enumerate(ca_result) if i not in nan_indices]  # CA 결과에서 제거
        final_effPoints = [value for i, value in enumerate(effPoints) if i not in nan_indices]  # CA 결과에서 제외된 effect 제거

    else:  # nan이 포함되지 않은 경우

        final_ca_result = ca_result
        final_effPoints = effPoints

    ecPoints_pred.append(final_ca_result)

result_df_pred = pd.DataFrame(ecPoints_pred, columns=effect, index=name_list)
final_df_pred = result_df_pred.transpose()


###################
##MDR calculation##
###################


## (Mean MDR) 전 구간에 대한 평가 ##

full_curve_mdr_list = []

for name in name_list:
    conc_obs = final_df_obs[name].tolist()
    conc_pred = final_df_pred[name].tolist()
    effect = [10, 20, 30, 40, 50, 60, 70, 80, 90]

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



result_df_mdr = pd.DataFrame(full_curve_mdr_list, index=name_list)




## (Point MDR) EC10, 50 기준 평가 ##
point_mdr_10_list = []
point_mdr_50_list = []

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
    point_mdr_50_list.append(conc_pred_F2[4] / conc_obs_F2[4])


## MDR 비교 데이터 표 생산 및 저장 ##
mdr_df = pd.DataFrame({
    'Point MDR (EC10)': point_mdr_10_list,
    'Point MDR (EC50)': point_mdr_50_list,
    'Full curve MDR': full_curve_mdr_list

})

output_file_mdr = output_file_path_mdr + "mdr_result.xlsx"

mdr_df.to_excel(output_file_mdr)









