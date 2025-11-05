# -*- coding: utf-8 -*-
"""
Created on 2025/11/5 15:36

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
from collections import defaultdict

DATA_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw"

stats = defaultdict(lambda: {"Co": set(), "Pt": set(), "Co_trials": 0, "Pt_trials": 0})

for f in os.listdir(DATA_DIR):
    if not f.endswith(".txt"): continue
    exp = f[:2]  # Ga / Ju / Si
    group = "Co" if "Co" in f else "Pt"
    subj = f[:6]
    stats[exp][group].add(subj)
    if group == "Co": stats[exp]["Co_trials"] += 1
    else: stats[exp]["Pt_trials"] += 1

for exp, v in stats.items():
    n_co = len(v["Co"])
    n_pt = len(v["Pt"])
    print(f"{exp}: {n_co} controls ({v['Co_trials']} trials), {n_pt} patients ({v['Pt_trials']} trials)")
