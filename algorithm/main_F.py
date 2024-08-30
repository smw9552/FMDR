# -------------------------------
# Script Name: main.py
# Author: Myungwon Seo
# Affiliation: Korea Research Institute of Chemical Technology (KRICT)
# Date: August 30, 2024 (Last updated)
# Version: 1.0
# ------------------------------
# Description:
# This script calculates FMDR of toxicity data from a given file path provided by the user.
# The user must specify the file path to the CSV file containing the data.
#
# Input:
# - file_path: A string representing the full path to the CSV or XLSL file with experimental data.
# - Example: "C:/Users/YourName/Documents/data/input_data.csv"
#
# Output:
# results.xlsx: A XLSX file containing the processed analysis results.
#
# Requirements:
# - Python 3.11
# - numpy library version 1.24.3
# - pandas library version 2.0.3
# - scipy library version 1.11.1
# -------------------------------



import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from algorithm.models import models
import random
import tensorflow as tf

m = models()


# Setting the seed
random_seed = 2024
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)
tf.random.set_seed(random_seed)


# load file info

# Please insert information related to the mixed toxicity DRC experimental data.
input_file_obs = "Please enter the path to the observed data file"
# (e.g., "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\test\\mixture_drc\\mixture_dataset_sample.xlsx")

# Please enter the information of mixtures containing data for each individual substance.
input_file_dir = "Please enter the directory to the single chemical drc files"
# (e.g., "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\test\\single_chemical_drc\\")

# Please enter the path to save the image results.
output_file_path_image = "Please enter the directory to save image results"
# (e.g., "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\test\\result\\mdr_graph\\")


# Please enter the path to save the FMDR results.
output_file_path_fmdr = "Please enter the path to save result of FMDR"
# (e.g., "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\test\\result\\")

# Please enter the path to save the MDR results. (Optional)
output_file_path_mdr = "Please enter the path to save result of conventional MDR"
# (e.g., "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\test\\result\\")



# Importing the script for the nonlinear function
module_path = "Please enter the file path where the script is located."
# (e.g., "C:\\Users\\user\\PycharmProjects\\FMDR\\algorithm\\")
rm_module_name = 'rm_python_opt'
exec(open(module_path + str("rm_python_opt.py"), "r", encoding='utf-8').read())



# Load batch file list
dir_filename = os.listdir(input_file_dir)
files = [file for file in dir_filename if file.endswith(".csv")]
file_list = []

for ai in range(0, len(files)):
    file_list.append(input_file_dir + files[ai])



# Sorting the file list numbers in order
def sort_files(file_list):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return float('inf')

    sorted_files = sorted(file_list, key=extract_number)

    return sorted_files

sorted_files = sort_files(file_list)
print(sorted_files)



# Extracting file names
sorted_file_name = []
for file in sorted_files:
    temp_file = str(file).replace(str(input_file_dir), "")
    sorted_file_name.append(temp_file.replace(".csv", ""))



# Setting toxicity effect intervals and the number of data points
effect = np.linspace(0.1, 0.9, 30) #default condition



# Calculating experimental data
ecPoints_obs, name_list = m.exp_model(input_file_obs, module_path, rm_module_name, effect)
result_final_df_obs = pd.DataFrame(ecPoints_obs, columns=effect, index=name_list)
final_df_obs = result_final_df_obs.transpose()



# Prediction using the CA model
ecPoints_ca = []
for bi in range(0, len(sorted_files)):
    ca = m.ca_model(sorted_files[bi], module_path, rm_module_name, random_seed, effect)
    ecPoints_ca.append(ca)

result_df_pred = pd.DataFrame(ecPoints_ca, columns=effect, index=name_list)
final_df_pred = result_df_pred.transpose()



# Calculating FMDR
full_curve_mdr_list = []

for name in name_list:
    conc_obs = final_df_obs[name].tolist()
    conc_pred = final_df_pred[name].tolist()

    # Checking for missing values (NaN) based on obs data
    if np.isnan(conc_obs).any():
        nan_indices_obs = np.where(np.isnan(conc_obs))[0]
        conc_obs_F1 = [value for i, value in enumerate(conc_obs) if i not in nan_indices_obs]
        conc_pred_F1 = [value for i, value in enumerate(conc_pred) if i not in nan_indices_obs]
        effect_F1 =  [value for i, value in enumerate(effect) if i not in nan_indices_obs]

    else:
        conc_obs_F1 = conc_obs
        conc_pred_F1 = conc_pred
        effect_F1 = effect

    # Checking for missing values (NaN) based on pred data
    if np.isnan(conc_pred).any():
        nan_indices_pred = np.where(np.isnan(conc_pred))[0]
        conc_obs_F2 = [value for i, value in enumerate(conc_obs_F1) if i not in nan_indices_pred]
        conc_pred_F2 = [value for i, value in enumerate(conc_pred_F1) if i not in nan_indices_pred]
        effect_F2 = [value for i, value in enumerate(effect_F1) if i not in nan_indices_pred]

    else:
        conc_obs_F2 = conc_obs_F1
        conc_pred_F2 = conc_pred_F1
        effect_F2 = effect_F1

    # Perform MDR calculation using the remaining data after excluding missing values
    temp_mdr_list = []
    for di in range(0, len(conc_obs_F2)): #conc_pred_F2와 사이즈 동일함
        temp_mdr_list.append(conc_pred_F2[di] / conc_obs_F2[di])

    full_curve_mdr_list.append(np.mean(temp_mdr_list)) #평균 mdr 계산 결과를 추가

    # Plotting the graph and saving it
    plt.plot(conc_obs_F2, effect_F2, '-*', label="Obs", color='black')
    plt.plot(conc_pred_F2, effect_F2, '-*', label="Pred (CA model)", color='blue')

    plt.title(name)
    plt.xlabel('Concentration')
    plt.ylabel('Effect(%)')
    plt.legend()

    output_file = output_file_path_image + str(name) + "_result.jpg"
    plt.savefig(output_file)
    plt.clf()



# Constructing a DataFrame with the FMDR results
fmdr_df = pd.DataFrame({
    'name': name_list,
    'Full curve MDR': full_curve_mdr_list

})



## File path and writing
output_file_fmdr = output_file_path_fmdr + "fmdr_result.xlsx"
fmdr_df.to_excel(output_file_fmdr, index=False)



# MDR calculation section to be used if needed
# Please remove the comments to use this MDR calculation section
'''

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
'''