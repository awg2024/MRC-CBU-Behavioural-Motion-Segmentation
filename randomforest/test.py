import pandas as pd
import df 

# Your CSV paths
merge_1 = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_12_Day3\script_outputs\merged_data.csv"
merge_2 = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_14_Day4\script_outputs\merged_data.csv"
merge_3 = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_08_Day1\script_outputs\merged_data.csv"
merge_4 = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_15_Day1\script_outputs\merged_data.csv"
merge_5 = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\merged_data.csv"

# Load them
df1 = pd.read_csv(merge_1, low_memory=False)
df2 = pd.read_csv(merge_2, low_memory=False)
df3 = pd.read_csv(merge_3, low_memory=False)
df4 = pd.read_csv(merge_4, low_memory=False)
df5 = pd.read_csv(merge_5, low_memory=False)

# Function to compare two DataFrames' columns
def compare_columns(df_a, df_b, name_a, name_b):
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)


    
    print(f"\n🔍 Comparing {name_a} vs {name_b}:")
    print(f"  Total in {name_a}: {len(cols_a)} | Total in {name_b}: {len(cols_b)}")
    print(f"  Missing in {name_a}: {cols_b - cols_a}")
    print(f"  Missing in {name_b}: {cols_a - cols_b}")
    print(f"  Columns in both: {len(cols_a & cols_b)}")


# Compare all pairs
compare_columns(df1, df2, "merge_1", "merge_2")
compare_columns(df1, df2, "merge_2", "merge_3")
compare_columns(df1, df2, "merge_1", "merge_3")
compare_columns(df1, df2, "merge_1", "merge_4")
compare_columns(df1, df2, "merge_4", "merge_2")
compare_columns(df1, df2, "merge_1", "merge_3")
compare_columns(df1, df2, "merge_1", "merge_5")


clf, saved_feature_cols = joblib.load("model.pkl")
current_cols = df.columns.tolist()

print("Columns in model feature list:", len(saved_feature_cols))
print("Columns in current data:", len(current_cols))
print("Missing from current data:", set(saved_feature_cols) - set(current_cols))
print("Extra in current data:", set(current_cols) - set(saved_feature_cols))

