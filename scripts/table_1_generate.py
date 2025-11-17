import pandas as pd
import re

# ====== Load summary CSV ======
csv_path = "/Users/wangyulin/Time Series/results_calvocnn_multi/results_calvocnn_all_summary.csv"
df = pd.read_csv(csv_path)

# Convert to numeric
df['acc'] = pd.to_numeric(df['acc'], errors='coerce')
df['auc'] = pd.to_numeric(df['auc'], errors='coerce')

# ====== Parse experiment string ======
def parse_exp(exp):
    """
    heatmaps_rawphase_both_σ10_i4000_Ga_dual_balanced
    """
    parts = exp.split('_')
    method   = parts[1]
    foot     = parts[2]
    sigma    = re.findall(r'σ(\d+)', exp)[0]
    interp   = re.findall(r'i(\d+)', exp)[0]
    dataset  = parts[5]
    taskmode = parts[6]
    return method, foot, dataset, sigma, interp, taskmode

df[['method','foot','dataset','sigma','interp','taskmode']] = \
    df['experiment'].apply(lambda x: pd.Series(parse_exp(x)))


# ====== Group only by (dataset × foot × taskmode) ======
best_rows = []
for keys, group in df.groupby(["dataset","foot","taskmode"]):
    g = group.sort_values(["auc","acc"], ascending=[False, False])
    best_rows.append(g.iloc[0])

best_df = pd.DataFrame(best_rows)

# ====== Keep columns for Table 1 ======
best_df = best_df[[
    'dataset','foot','taskmode',
    'method','sigma','interp',
    'mode','fc_dim','acc','auc'
]]

# ====== Sort for readability ======
best_df = best_df.sort_values(
    by=['dataset','taskmode','foot'],
    ascending=[True, True, True]
)

# ====== Save final minimal Table 1 ======
out_csv = "/Users/wangyulin/Time Series/results_calvocnn_multi/results_calvocnn_table1_minimal.csv"
best_df.to_csv(out_csv, index=False)

print("\n===== FINAL MINIMAL TABLE 1 =====\n")
print(best_df)
print("\nSaved to:", out_csv)
