# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:21

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import pandas as pd

df = pd.read_csv("/data/labels.csv")
print(df['label'].value_counts())
