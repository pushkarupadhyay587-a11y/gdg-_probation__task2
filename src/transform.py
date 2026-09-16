import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from pathlib import Path
import joblib
csv_path = Path(__file__).resolve().parents[1] / "notebooks" / "data" / "processed" / "modified_data.csv"

df = pd.read_csv(csv_path)

df['Will_Buy_EV'] = df['Will_Buy_EV'].map({"Yes":1,"No":0})

cat_cols = ['Gender', 'City_Type', 'Current_Car_Type','Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level'
]
num_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned','Charging_Stations_Near_Home','Charging_Stations_Near_Work',
'Environmental_Concern_Level']


preprocessor = ColumnTransformer(
    transformers = [
        ('cat',OneHotEncoder(),cat_cols),
        ('num',StandardScaler(),num_cols)
    ])

df_processed = preprocessor.fit_transform(df)

cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
num_feature_names = num_cols
all_feature_names = list(cat_feature_names) + num_feature_names 

df_processed = pd.DataFrame(df_processed.toarray() if hasattr(df_processed, "toarray") else df_processed,columns=all_feature_names)
df_processed['Will_Buy_EV'] = df['Will_Buy_EV']

final_csv_path = csv_path = Path(__file__).resolve().parents[1] / "notebooks" / "data" / "processed" / "scaled_data.csv"

df_processed.to_csv(final_csv_path,index=False)

