import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
df = pd.read_csv('data/a2d2/csv_backup/features_train (copy).csv')

target_labels = ['Car', 'Truck', 'Pedestrian']
df = df[df['label'].isin(target_labels)]

# 2. 수치형 컬럼 선택 ('frame_id' 제외)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'frame_id' in numeric_cols:
    numeric_cols.remove('frame_id')

# 3. 서브플롯 설정
num_plots = len(numeric_cols)
cols_per_row = 3  
rows = (num_plots // cols_per_row) + (1 if num_plots % cols_per_row > 0 else 0)

plt.figure(figsize=(20, 6 * rows))

# 4. 각 컬럼별로 클래스(label)에 따른 히스토그램 그리기
for i, col in enumerate(numeric_cols):
    ax = plt.subplot(rows, cols_per_row, i + 1) # ax 변수로 받기
    
    sns.histplot(
        data=df, 
        x=col, 
        hue='label', 
        kde=True, 
        element="step", 
        bins=30,
        stat="density",      
        common_norm=False    
    )
    
    plt.title(f'{col}', fontsize=18)
    plt.xlabel(col, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    
    # --- [핵심 수정] 범례(Legend) 크기 조절 ---
    # 현재 그려진 그래프(ax)에서 범례 객체를 가져옴
    legend = ax.get_legend()
    
    if legend:
        # 1. 범례 제목 ('label') 크기 조절
        plt.setp(legend.get_title(), fontsize=18) 
        # 2. 범례 항목 텍스트 ('Car', 'Truck' 등) 크기 조절
        plt.setp(legend.get_texts(), fontsize=18)
    # ----------------------------------------

plt.tight_layout()
plt.show()