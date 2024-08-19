import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_file_path = 'C:\\Users\\user\\PycharmProjects\\FMDR\\data\\graph_input\\'
input_file_name_cmdr_fmdr = 'input_cmdr-fmdr.xlsx'

output_file_path = 'C:\\Users\\user\\PycharmProjects\\FMDR\\data\\graph_output\\'
output_file_name_cmdr_fmdr = 'output_cmdr_fmdr.tiff'


# 데이터프레임 정의 (이전에 데이터를 로드한 부분을 포함)
file_path_comparison = str(input_file_path) + str(input_file_name_cmdr_fmdr)
df_comparison = pd.read_excel(file_path_comparison, sheet_name='Sheet1')


# Combine Point MDR and FMDR for violin plot
melted_data = df_comparison.melt(id_vars=['Name'],
                                 value_vars=['CMDR (EC10)', 'CMDR (EC30)', 'CMDR (EC50)', 'CMDR (EC70)',
                                             'FMDR (10 points)', 'FMDR (30 points)', 'FMDR (50 points)',
                                             'FMDR (100 points)', 'FMDR (1000 points)'],
                                 var_name='Type', value_name='MDR')


# 범주 이름을 아래첨자를 사용하여 수정 (특히 EC10에 대해)
melted_data['Type'] = melted_data['Type'].replace({
    'CMDR (EC10)': 'CMDR (EC$_{10}$)',  # EC10을 아래첨자로 표시
    'CMDR (EC30)': 'CMDR (EC$_{30}$)',
    'CMDR (EC50)': 'CMDR (EC$_{50}$)',
    'CMDR (EC70)': 'CMDR (EC$_{70}$)',
    'FMDR (10 points)': 'FMDR (10 points)',
    'FMDR (30 points)': 'FMDR (30 points)',
    'FMDR (50 points)': 'FMDR (50 points)',
    'FMDR (100 points)': 'FMDR (100 points)',
    'FMDR (1000 points)': 'FMDR (1000 points)'
})



colors = ['b', 'g', 'r', 'c', 'm', 'pink', 'lime', 'purple', 'orange']


# Draw Violin Plot #
plt.figure(figsize=(12, 7))
sns.violinplot(x='Type', y='MDR', data=melted_data, palette=colors, inner='quartile', )

plt.xlabel('Types of approaches', fontsize=12)
plt.ylabel('MDR value', fontsize=12)
plt.title('Comparison of CMDR and FMDR approaches', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7)

plt.tight_layout()


# 그래프 출력
#plt.show()

# 그래프 저장
plt.savefig(str(output_file_path) + str(output_file_name_cmdr_fmdr), dpi=300)