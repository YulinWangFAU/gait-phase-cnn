import os
import re
import pandas as pd

# === 1️⃣ 设置路径 ===
base_dir = '/Users/wangyulin/Time Series/'
folders = [
    'results_cnn_balanced_g8_i2000',
    'results_cnn_balanced_g10_i4000',
    'results_cnn_balanced_g12_i5000',
    'results_resnet_focal_balanced_g8_i2000',
    'results_resnet_focal_balanced_g10_i4000',
    'results_resnet_focal_balanced_g12_i5000'
]

all_data = []

# === 2️⃣ 遍历每个实验文件夹 ===
for folder in folders:
    folder_path = os.path.join(base_dir, folder)

    # 递归查找所有 .csv 文件
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.csv'):
                csv_path = os.path.join(root, f)
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"⚠️ 无法读取 {csv_path}: {e}")
                    continue

                # 如果没有 experiment 列，跳过
                if 'experiment' not in df.columns:
                    continue

                # === 提取文件夹元信息 ===
                model = 'CNN' if 'cnn' in folder.lower() else 'ResNet'
                loss = 'CrossEntropy' if 'cnn' in folder.lower() else 'FocalLoss'
                sigma_match = re.search(r'g(\d+)', folder)
                interp_match = re.search(r'i(\d+)', folder)
                sigma = int(sigma_match.group(1)) if sigma_match else None
                interp = int(interp_match.group(1)) if interp_match else None

                # === 从 experiment 中提取组别 (Ga/Ju/Si) ===
                def extract_group(x):
                    m = re.search(r'_(Ga|Ju|Si)_', str(x))
                    return m.group(1) if m else 'Unknown'

                df['group'] = df['experiment'].apply(extract_group)

                # 添加模型和参数信息
                df['model'] = model
                df['loss'] = loss
                df['sigma'] = sigma
                df['interp'] = interp

                all_data.append(df)

# === 3️⃣ 合并数据 ===
if not all_data:
    raise ValueError("❌ 未找到任何 CSV 文件，请检查路径或文件名。")

merged_df = pd.concat(all_data, ignore_index=True)

# === 4️⃣ 提取关键指标 ===
metrics = ['acc', 'auc', 'f1_Co', 'f1_Pt']
meta_cols = ['model', 'loss', 'sigma', 'interp', 'group']
merged_df = merged_df[meta_cols + metrics]

# === 5️⃣ 计算总体与分组平均 ===
# 总体平均（跨组别）
overall_summary = (
    merged_df.groupby(['model', 'loss', 'sigma', 'interp'])[metrics]
    .mean()
    .reset_index()
    .sort_values(['model', 'sigma'])
)

# 分组平均（按 Ga/Ju/Si）
groupwise_summary = (
    merged_df.groupby(['model', 'loss', 'sigma', 'interp', 'group'])[metrics]
    .mean()
    .reset_index()
    .sort_values(['model', 'group', 'sigma'])
)

# === 6️⃣ 导出结果 ===
overall_path = os.path.join(base_dir, 'all_results_summary.csv')
groupwise_path = os.path.join(base_dir, 'groupwise_results_summary.csv')

overall_summary.to_csv(overall_path, index=False)
groupwise_summary.to_csv(groupwise_path, index=False)

print(f"✅ 汇总完成：\n  - 总体平均：{overall_path}\n  - 分组平均：{groupwise_path}")

# 打印部分结果预览
print("\n=== 总体平均 ===")
print(overall_summary.head())
print("\n=== 分组平均 ===")
print(groupwise_summary.head())
