import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def map_icd9(code):
    if pd.isnull(code) or str(code).strip() == '?':
        return 'Other'
    code_str = str(code).strip()
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
    try:
        val = float(code_str)
        if 390 <= val <= 459 or val == 785:
            return 'Circulatory'
        elif 460 <= val <= 519 or val == 786:
            return 'Respiratory'
        elif 520 <= val <= 579 or val == 787:
            return 'Digestive'
        elif 250 <= val < 251:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        if code_str.startswith('250'):
            return 'Diabetes'
        return 'Other'

def preprocess_data(data_path="data/diabetic_data.csv"):
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Initial shape: {df.shape}")
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # 1. Keep only the first encounter for each patient to avoid duplicate patient bias
    print("Removing multiple encounters for the same patient (keeping first encounter)...")
    df = df.drop_duplicates(subset='patient_nbr', keep='first')
    print(f"Shape after keeping first encounters: {df.shape}")
    
    # 2. Exclude deceased patients and those discharged to hospice
    # Exclude IDs: 11 (Expired), 13 (Hospice / home), 14 (Hospice / medical facility), 
    # 19 (Expired at home), 20 (Expired in facility), 21 (Expired, place unknown)
    print("Excluding deceased and hospice patients...")
    exclude_disposition_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(exclude_disposition_ids)]
    print(f"Shape after removing deceased/hospice: {df.shape}")
    
    # 3. Create target variable: Readmitted < 30 days -> 1 (High Risk), else -> 0
    # Map readmitted: '<30' -> 1, '>30' -> 0, 'NO' -> 0
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    print(f"Class distribution of readmitted_binary:\n{df['readmitted_binary'].value_counts(normalize=True)}")
    
    # 4. Drop columns with excessive missing values (>40%)
    # weight (~97% missing), payer_code (~40% missing), medical_specialty (~49% missing)
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr', 'readmitted']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 5. Preprocess patient demographics
    # Map age intervals to midpoints (e.g. '[0-10)' -> 5, '[10-20)' -> 15, etc.)
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    df['age_num'] = df['age'].map(age_mapping)
    df = df.drop(columns=['age'])
    
    # Handle gender: Remove rows where gender is Unknown/Invalid, map Male -> 1, Female -> 0
    df = df[df['gender'].isin(['Male', 'Female'])]
    df['gender_binary'] = df['gender'].map({'Male': 1, 'Female': 0})
    df = df.drop(columns=['gender'])
    
    # Impute missing race values as 'Unknown'
    df['race'] = df['race'].fillna('Unknown')
    
    # 6. Feature engineering: Healthcare utilization
    # Sum total prior visits
    df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    
    # 7. Feature engineering: Clinical diagnostics
    # Map ICD-9 diagnoses codes to 9 clinical categories
    df['diag_1_cat'] = df['diag_1'].apply(map_icd9)
    df['diag_2_cat'] = df['diag_2'].apply(map_icd9)
    df['diag_3_cat'] = df['diag_3'].apply(map_icd9)
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
    
    # 8. Feature engineering: Medications and medication changes
    medication_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    
    # Count total generic medications prescribed to the patient (i.e. value is not 'No')
    df['num_meds_prescribed'] = df[medication_cols].apply(lambda row: sum(1 for val in row if val != 'No'), axis=1)
    
    # Count number of medication dosage adjustments (i.e. value is 'Up' or 'Down')
    df['num_med_changes'] = df[medication_cols].apply(lambda row: sum(1 for val in row if val in ['Up', 'Down']), axis=1)
    
    # Map general 'change' and 'diabetesMed' features to binary
    df['change_binary'] = df['change'].map({'Ch': 1, 'No': 0})
    df['diabetesMed_binary'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})
    df = df.drop(columns=['change', 'diabetesMed'])
    
    # One-hot encode the 23 medication columns to represent their state
    df = pd.get_dummies(df, columns=medication_cols, drop_first=True, dtype=int)
    
    # 9. Handle A1Cresult and max_glu_serum
    df['A1Cresult'] = df['A1Cresult'].fillna('None')
    df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
    
    # Convert categorical variables to string categories for dummy encoding
    categorical_cols = [
        'race', 'max_glu_serum', 'A1Cresult', 
        'diag_1_cat', 'diag_2_cat', 'diag_3_cat',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    # Dummy encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    print(f"Final preprocessed dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 10. Split into features and target
    X = df.drop(columns=['readmitted_binary'])
    y = df['readmitted_binary']
    
    # 11. Train-test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 12. Standard scale continuous features
    continuous_features = [
        'age_num', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'num_meds_prescribed', 'num_med_changes', 'total_prior_visits'
    ]
    
    scaler = StandardScaler()
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])
    
    # Save the processed split datasets to csv files in data/
    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    print("Preprocessed files successfully saved to 'data/' folder.")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
