import pandas as pd
import matplotlib.pyplot as plt


# Point MDR #

# 데이터 로드
df = pd.read_excel('C:\\Users\\user\\PycharmProjects\\FMDR\\data\\graph_input\\point_mdr_input.xlsx', sheet_name='Sheet1')

# 그래프를 그리기 위한 준비
x_labels = df['name']
ec_values = ['Point MDR (EC10)', 'Point MDR (EC30)', 'Point MDR (EC50)', 'Point MDR (EC70)']
colors = ['b', 'g', 'r', 'c']  # EC10, EC30, EC50, EC70에 대한 색상

# 개선된 그래프 그리기
plt.figure(figsize=(12, 7))

for i, ec in enumerate(ec_values):
    plt.plot(x_labels, df[ec], marker='o', markersize=8, markeredgewidth=2,
             markeredgecolor='black', color=colors[i], linewidth=2, label=ec)

plt.xlabel('Name', fontsize=14)
plt.ylabel('MDR', fontsize=14)
plt.title('MDR Values for Each Mix', fontsize=16)
plt.legend(title='EC Levels', title_fontsize='13', fontsize='12', loc='upper right')

# 축 라벨 가독성 개선
plt.xticks(rotation=60, fontsize=12, ha='right')
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tight_layout()

# 그래프 출력
plt.show()




# FMDR #
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df_updated = pd.read_excel('C:\\Users\\user\\PycharmProjects\\FMDR\\data\\graph_input\\fmdr_input.xlsx', sheet_name='Sheet1')

# 그래프를 그리기 위한 준비
x_labels_updated = df_updated['name']
fmdr_values_updated = ['FMDR (10 points)', 'FMDR (30 points)', 'FMDR (50 points)',
                       'FMDR (100 points)', 'FMDR (1000 points)']

# 새로운 색상 조합 (보라색 대신 주황색)
colors_alternative = ['b', 'g', 'r', 'c', 'orange']  # 보라색을 주황색으로 대체

# 그래프 그리기
plt.figure(figsize=(12, 7))

for i, fmdr in enumerate(fmdr_values_updated):
    plt.plot(x_labels_updated, df_updated[fmdr], marker='o', markersize=8, markeredgewidth=2,
             markeredgecolor='black', color=colors_alternative[i], linewidth=2, label=fmdr)

# 라벨링
plt.xlabel('Name', fontsize=14)
plt.ylabel('FMDR', fontsize=14)
plt.title('FMDR Values for Each Mix', fontsize=16)

# y축 범위 설정
plt.ylim(0, 2.5)

# 범례 설정
plt.legend(title='FMDR Points', fontsize='12', loc='upper right')

# 축 라벨 및 틱 설정
plt.xticks(rotation=60, fontsize=12, ha='right')
plt.yticks(fontsize=12)

# 그리드 및 레이아웃 설정
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tight_layout()

# 그래프 출력
plt.show()