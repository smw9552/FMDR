from matplotlib.ticker import FuncFormatter
import importlib
import sys
import os
import random
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import numpy as np
os.environ["THREADPOOLCTL_THREAD_COUNT"] = "1"


class models:

    def exp_model(self, input_file, module_path, rm_module_name, effects):

        # rm_python을 모듈로 정의하여 함수 불러오기
        if module_path not in sys.path:
            sys.path.append(module_path)

        try:
            rm_module = importlib.import_module(rm_module_name)
            # print(f"Module '{rm_module_name}' imported successfully")
        except ModuleNotFoundError as e:
            print(f"Error: {e}")

        df = pd.read_excel(input_file)
        # df = pd.read_csv(input_file)

        name_list = df['name']
        model_list = df['RM']
        param_a_list = df['a']
        param_b_list = df['b']
        param_c_list = df['g']
        ec50_list = df['EC50']

        # scale check (fraction or percentage)
        # 단일물질 기준 DRC 정보를 활용하기 때문에 scale에 따라 모두 적용 필요
        ec_check_list = []
        for i in range(0, df.shape[0]):
            # print(model_list[i])
            rm_func = getattr(rm_module, str(model_list[i]))  # 동적 모듈 가져와서 함수 호출
            # rm_func = globals().get(model_list[i])  # 함수명으로 전역 네임스페이스에서 검색하여 객체를 가져옴
            # print(rm_func)

            if str(param_c_list[i]) != 'nan':  # 파라미터가 3개인 경우
                ec_check_list.append(
                    rm_func(ec50_list[i], param_a_list[i], param_b_list[i], param_c_list[i], mode='ce'))
            else:  # 파라미터가 2개인 경우
                ec_check_list.append(rm_func(ec50_list[i], param_a_list[i], param_b_list[i], mode='ce'))

        if sum(ec_check_list) > df.shape[0]:
            # percentage scale (in vitro 실험데이터는 percentage scale에 가까움)
            print("##Percentage scale##")
            scale = "Percentage"
        else:
            # fraction scale
            print("##Fraction scale##")
            scale = "Fraction"

        # 각 물질별 함수를 이용한 effect에 대한 농도값 계산
        # parameters: alpha(height), beta(slope), gamma(center point)
        # y축 scale 조정을 위해서는 DRC parameter 중 alpha(height)를 조정해야함
        # (중요) (1) Percentage -> Fraction: alpha / 100, (2) Fraction -> Percentage: alpha * 100
        ecPoints = []
        # scale을 fraction으로 고정하므로 effPoint도 fraction에 맞게 고정
        effPoints = effects
        # effPoints = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  ## <--- 10개 구간이 아니라 구간을 더 증가시킬 필요가 있음 ##
        # effPoints = [0.025, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.47, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]

        for ai in range(0, df.shape[0]):
            rm_func = getattr(rm_module, str(model_list[ai]))  # 동적 모듈 가져와서 함수 호출
            # rm_func = globals().get(model_list[ai])  # 함수명으로 전역 네임스페이스에서 검색하여 객체를 가져옴
            # print(rm_func)
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
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], param_c_list[ai],
                                    mode='ec'))
                    else:
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], mode='ec'))
                # rm_func(y, alpha, beta, gamma) - 파라미터 3개 이상인 경우 따로 만들어야 함(추가 제어문 설정 필요)

            ecPoints.append(temp_ecPoints)

        return ecPoints, name_list

    def ca_model(self, input_file, module_path, rm_module_name, random_seed, effects):

        random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        tf.random.set_seed(random_seed)

        # rm_python을 모듈로 정의하여 함수 불러오기
        if module_path not in sys.path:
            sys.path.append(module_path)

        try:
            rm_module = importlib.import_module(rm_module_name)
            # print(f"Module '{rm_module_name}' imported successfully")
        except ModuleNotFoundError as e:
            print(f"Error: {e}")

        # rm_module = importlib.import_module(rm_module_name)

        df = pd.read_csv(input_file)

        name_list = df['Substance']
        cas_list = df['CAS NO']
        comp_list = df['percentage']
        model_list = df['RM']
        input_unit_list = df['Unit for RM']
        output_unit_list = df['Coversion Unit']  # Coversion unit을 Conversion Unit으로 수정해야함
        param_a_list = df['a']
        param_b_list = df['b']
        param_c_list = df['g']
        ec50_list = df['EC50']

        # scale check (fraction or percentage)
        # 단일물질 기준 DRC 정보를 활용하기 때문에 scale에 따라 모두 적용 필요
        ec_check_list = []
        for i in range(0, df.shape[0]):
            # print(model_list[i])
            rm_func = getattr(rm_module, str(model_list[i]))  # 동적 모듈 가져와서 함수 호출
            # rm_func = globals().get(model_list[i])  # 함수명으로 전역 네임스페이스에서 검색하여 객체를 가져옴
            # print(rm_func)

            if str(param_c_list[i]) != 'nan':  # 파라미터가 3개인 경우
                ec_check_list.append(
                    rm_func(ec50_list[i], param_a_list[i], param_b_list[i], param_c_list[i], mode='ce'))
            else:  # 파라미터가 2개인 경우
                ec_check_list.append(rm_func(ec50_list[i], param_a_list[i], param_b_list[i], mode='ce'))

        if sum(ec_check_list) > df.shape[0]:
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
        effPoints = effects
        # effPoints = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  ## <--- 10개 구간이 아니라 구간을 더 증가시킬 필요가 있음 ##
        # effPoints = [0.025, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.47, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99]

        for ai in range(0, df.shape[0]):
            rm_func = getattr(rm_module, str(model_list[ai]))  # 동적 모듈 가져와서 함수 호출
            # rm_func = globals().get(model_list[ai])  # 함수명으로 전역 네임스페이스에서 검색하여 객체를 가져옴
            # print(rm_func)
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
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], param_c_list[ai],
                                    mode='ec'))
                    else:
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], mode='ec'))
                # rm_func(y, alpha, beta, gamma) - 파라미터 3개 이상인 경우 따로 만들어야 함(추가 제어문 설정 필요)

            ecPoints.append(temp_ecPoints)

        # Effect 결과를 계산하기 위해 CA 계산을 통해 농도구간 산출
        cal_ecPoints = []
        # 개별물질 함량 값을 effPoints의 농도값으로 나누는 과정
        for ci in range(0, df.shape[0]):
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
            final_effPoints = [value for i, value in enumerate(effPoints) if
                               i not in nan_indices]  # CA 결과에서 제외된 effect 제거

        else:  # nan이 포함되지 않은 경우

            final_ca_result = ca_result
            final_effPoints = effPoints

        print("CA prediction")
        print(final_ca_result)

        return final_ca_result