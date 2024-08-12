import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터프레임 정의 (이전에 데이터를 로드한 부분을 포함)
file_path_comparison = "C:\\Users\\user\\PycharmProjects\\FMDR\\data\\graph_input\\point_mdr-fmdr_input.xlsx"
df_comparison = pd.read_excel(file_path_comparison, sheet_name='Sheet1')


# Combine Point MDR and FMDR for violin plot
melted_data = df_comparison.melt(id_vars=['name'],
                                 value_vars=['Point MDR (EC10)', 'Point MDR (EC30)', 'Point MDR (EC50)', 'Point MDR (EC70)',
                                             'FMDR (10 points)', 'FMDR (30 points)', 'FMDR (50 points)',
                                             'FMDR (100 points)', 'FMDR (1000 points)'],
                                 var_name='Type', value_name='MDR')

# Draw Violin Plot #
plt.figure(figsize=(12, 7))
sns.violinplot(x='Type', y='MDR', data=melted_data, palette='Set2', inner='quartile')

plt.xlabel('Type')
plt.ylabel('MDR')
plt.title('Violin Plot comparing Point MDR and FMDR')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()