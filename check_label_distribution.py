# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:21
检查数据是否类别均衡
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("/data/labels.csv")
df = pd.read_csv("/Users/wangyulin/Time Series/629/gait-phase-cnn/data/labels.csv")
print(df['label'].value_counts())


# 绘制柱状图
counts = df['label'].value_counts().sort_index()
counts.index = ['Control (0)', 'Patient (1)']  # 给标签加名字

plt.figure(figsize=(6,4))
counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Label Distribution (Co vs Pt)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()