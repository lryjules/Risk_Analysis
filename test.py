import pandas as pd

df = pd.read_csv(
    "/Users/julesleroy/Desktop/Eseo/Année 2025-2026/PFE/bank_KM1.csv",
    sep=",",
    encoding="utf-8",
    engine="python"
)
print(df.head())