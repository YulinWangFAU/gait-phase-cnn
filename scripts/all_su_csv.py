# -*- coding: utf-8 -*-
"""
Created on 2025/11/11 11:31

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import re
import pandas as pd

# === 1️⃣ 设置你的基础路径 ===
base_dir = '/Users/wangyulin/Time Series/'

# 你要合并的所有实验文件夹
folders = [
    'results_cnn_balanced_g8_i2000',
    'results_cnn_balanced_g10_i4000',
    'results_cnn_balanced_g12_i5000',
    'results_resnet_focal_balanced_g8_i2000',
    'results_resnet_focal_balanced_g10_i4000',
    'results_resnet_focal_balanced_g12_i5000'
]

# === 2️⃣ 递归读取所有 CSV 文件 ===
all_dfs = []

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.endswith('.csv'):
                continue
            csv_path = os.path.join(root, f)

            # 尝试读取 CSV
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"⚠️ 无法读取 {csv_path}: {e}")
                continue

            # 必须包含 'experiment' 列，否则跳过
            if 'experiment' not in df.columns:
                continue

            # === 3️⃣ 从文件夹名提取模型和参数 ===
            model = 'CNN' if 'cnn' in folder.lower() else 'ResNet'
            loss = 'CrossEntropy' if 'cnn' in folder.lower() else 'FocalLoss'

            sigma_match = re.search(r'g(\d+)', folder)
            interp_match = re.search(r'i(\d+)', folder)
            sigma = int(sigma_match.group(1)) if sigma_match else None
            interp = int(interp_match.group(1)) if interp_match else None

            # === 4️⃣ 添加标注列 ===
            df['model'] = model
            df['loss'] = loss
            df['sigma'] = sigma
            df['interp'] = interp
            df['source_file'] = f

            all_dfs.append(df)

# === 5️⃣ 合并所有结果 ===
if not all_dfs:
    raise ValueError("❌ 没有找到任何有效 CSV 文件，请检查路径。")

merged_df = pd.concat(all_dfs, ignore_index=True)

# === 6️⃣ 输出结果 ===
save_path = os.path.join(base_dir, 'all_experiments_detailed.csv')
merged_df.to_csv(save_path, index=False)
print(f"✅ 已整合所有实验结果，共 {len(merged_df)} 条记录")
print(f"📁 已保存到: {save_path}")

# 打印前几行预览
print(merged_df.head())
