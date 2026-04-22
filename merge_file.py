import pandas as pd

file1 = "historical_price_4339.parquet"
file2 = "parquet_all1.parquet"
output_file = "parquet_all.parquet"

df1 = pd.read_parquet(file1)
df2 = pd.read_parquet(file2)

df_all = pd.concat([df1, df2], axis=0, ignore_index=True)

df_all.to_parquet(output_file, index=False)

print("Rows file1:", len(df1))
print("Rows file2:", len(df2))
print("Rows all:", len(df_all))