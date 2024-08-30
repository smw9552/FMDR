# -------------------------------
# Script Name: model.py
# Author: Myungwon Seo
# Affiliation: Korea Research Institute of Chemical Technology (KRICT)
# Date: August 30, 2024 (Last updated)
# Version: 1.0
# ------------------------------
# Description:
# This script is designed to analyze experimental DRC information and utilize the CA prediction model.
#
# Requirements:
# - Python 3.11
# - numpy library version 1.24.3
# - pandas library version 2.0.3
# - scipy library version 1.11.1
# -------------------------------



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
        ec_check_list = []
        for i in range(0, df.shape[0]):
            # print(model_list[i])
            rm_func = getattr(rm_module, str(model_list[i]))

            if str(param_c_list[i]) != 'nan':
                ec_check_list.append(
                    rm_func(ec50_list[i], param_a_list[i], param_b_list[i], param_c_list[i], mode='ce'))
            else:
                ec_check_list.append(rm_func(ec50_list[i], param_a_list[i], param_b_list[i], mode='ce'))

        if sum(ec_check_list) > df.shape[0]:
            # percentage scale
            print("##Percentage scale##")
            scale = "Percentage"
        else:
            # fraction scale
            print("##Fraction scale##")
            scale = "Fraction"


        ecPoints = []

        effPoints = effects

        for ai in range(0, df.shape[0]):
            rm_func = getattr(rm_module, str(model_list[ai]))
            temp_ecPoints = []

            if scale == "Fraction":
                for bi in range(0, len(effPoints)):
                    if str(param_c_list[ai]) != 'nan':
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], param_a_list[ai], param_b_list[ai], param_c_list[ai], mode='ec'))
                    else:
                        temp_ecPoints.append(rm_func(effPoints[bi], param_a_list[ai], param_b_list[ai], mode='ec'))

            else:  # percentage scale
                for bi in range(0, len(effPoints)):
                    if str(param_c_list[ai]) != 'nan':
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], param_c_list[ai],
                                    mode='ec'))
                    else:
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], mode='ec'))
            ecPoints.append(temp_ecPoints)

        return ecPoints, name_list

    def ca_model(self, input_file, module_path, rm_module_name, random_seed, effects):

        random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        tf.random.set_seed(random_seed)

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
        output_unit_list = df['Coversion Unit']
        param_a_list = df['a']
        param_b_list = df['b']
        param_c_list = df['g']
        ec50_list = df['EC50']

        # scale check (fraction or percentage)
        ec_check_list = []
        for i in range(0, df.shape[0]):
            # print(model_list[i])
            rm_func = getattr(rm_module, str(model_list[i]))

            if str(param_c_list[i]) != 'nan':
                ec_check_list.append(
                    rm_func(ec50_list[i], param_a_list[i], param_b_list[i], param_c_list[i], mode='ce'))
            else:
                ec_check_list.append(rm_func(ec50_list[i], param_a_list[i], param_b_list[i], mode='ce'))

        if sum(ec_check_list) > df.shape[0]:
            print("##Percentage scale##")
            scale = "Percentage"
        else:
            # fraction scale
            print("##Fraction scale##")
            scale = "Fraction"

        # composition scale check
        total_comp = sum(comp_list)
        if total_comp <= 1:
            new_comp_list = comp_list
        else:
            new_comp_list = []
            for j in range(0, len(comp_list)):
                new_comp_list.append(comp_list[j] / total_comp)


        ecPoints = []
        effPoints = effects


        for ai in range(0, df.shape[0]):
            rm_func = getattr(rm_module, str(model_list[ai]))
            temp_ecPoints = []

            if scale == "Fraction":
                for bi in range(0, len(effPoints)):
                    if str(param_c_list[ai]) != 'nan':
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], param_a_list[ai], param_b_list[ai], param_c_list[ai], mode='ec'))
                    else:
                        temp_ecPoints.append(rm_func(effPoints[bi], param_a_list[ai], param_b_list[ai], mode='ec'))

            else:  # percentage scale
                for bi in range(0, len(effPoints)):
                    if str(param_c_list[ai]) != 'nan':
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], param_c_list[ai],
                                    mode='ec'))
                    else:
                        temp_ecPoints.append(
                            rm_func(effPoints[bi], (param_a_list[ai] / 100), param_b_list[ai], mode='ec'))

            ecPoints.append(temp_ecPoints)

        cal_ecPoints = []
        for ci in range(0, df.shape[0]):
            cal_ecPoints.append([(new_comp_list[ci] / element) for element in ecPoints[ci]])

        sum_cal_ecPoints = [sum(sublist) for sublist in zip(*cal_ecPoints)]
        ca_result = [1 / element for element in sum_cal_ecPoints]

        # Checking for NaN values based on CA results
        if np.isnan(ca_result).any():  # nan이 포함된 경우
            # nan 위치 확인 및 nan 제거
            nan_indices = np.where(np.isnan(ca_result))[0]
            final_ca_result = [value for i, value in enumerate(ca_result) if i not in nan_indices]  # CA 결과에서 제거
            final_effPoints = [value for i, value in enumerate(effPoints) if
                               i not in nan_indices]

        else:

            final_ca_result = ca_result
            final_effPoints = effPoints

        print("CA prediction")
        print(final_ca_result)

        return final_ca_result