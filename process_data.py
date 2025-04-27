"""
Reading the data from GoogleDrive
"""

#Patients dataset
def load_csv_from_gdrive(file_id):
    # url for GoogleDrive
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    # Convert response text into a file-like object so pandas can read it
    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data)
    return df

##############################################
# List of States & fileID   (Patients)
##############################################
file_ids = {
    "PA": "1HPn-erywYpJgCmGseK7FKmwvcJzjEpTn",
    "IL": "1gn8ox4ugz6-917Myzl6o-rsWwYj5edab",
    "FL": "1R30A6cOrGveVkRhgmXemcPvlygqXuUfY",
    "TX": "18Eso9ZJhrwcN5Av0H7haZgQsgO2NOEQ8",
    "CA": "1uoFrXT2wQiBK63rOiVYxQs7NeiXAYhUR"
}

# List to store each state's DataFrame
df_list = []

# Loop over each state and file_id
for state, fid in file_ids.items():
    df_state = load_csv_from_gdrive(fid)
    df_list.append(df_state)

# Combine all DataFrames into one
df_combined = pd.concat(df_list, ignore_index=True)

#Convert BIRTHDATE column to a datetime type
df_combined["BIRTHDATE"] = pd.to_datetime(df_combined["BIRTHDATE"])
df_combined["DEATHDATE"] = pd.to_datetime(df_combined["DEATHDATE"])
df_combined["BIRTHDATE_ORD"] = df_combined["BIRTHDATE"].apply(lambda d: d.toordinal())

#Define a function that calculates age at death
def calc_age_at_death(row):
    birth = row["BIRTHDATE"]
    death = row["DEATHDATE"]
    if pd.isnull(death):
        return np.nan
    return (death - birth).days // 365
df_combined["AGE_AT_DEATH"] = df_combined.apply(calc_age_at_death, axis=1)

keep_columns = ["Id", "BIRTHDATE", "BIRTHDATE_ORD", "DEATHDATE", "AGE_AT_DEATH", "MARITAL",
                "RACE", "ETHNICITY", "GENDER", "CITY", "STATE", "COUNTY", "INCOME"]
patients_all = df_combined[keep_columns]

#do not output as they're too big.
#patients_all.to_csv("data/processed/patients_all.csv", index=False)


##############################################
# List of States & fileID   (Conditions)
##############################################

# List of States & fileID for conditions
file_ids = {
    "PA": "1xeuzrlJ8PeeV_UYAdKuS0-SWZP4iW_zc",
    "IL": "1697EYEVyhmkFTME7LMkJiMOFeZmapf4q",
    "FL": "1-HvfZFniBsjbnepdH4apquD3nXeQ8KlQ",
    "TX": "17zzvVVcXt8uhKr8_h9J-1MeDZAb1C5Pe",
    "CA": "1k4a4sKSHUyUjR2kQu-lObJC8ggpJ3Y3d"
}

# List to store each state's DataFrame
df_list = []
# Loop over each state and file_id
for state, fid in file_ids.items():
    df_state = load_csv_from_gdrive(fid)
    df_list.append(df_state)
# Combine all DataFrames into one
df_combined = pd.concat(df_list, ignore_index=True)

# FILTER *only* rows where DESCRIPTION is exactly "Myocardial infarction (disorder)"
df_infarction_only = df_combined[df_combined["DESCRIPTION"].str.strip() == "Myocardial infarction (disorder)"]
df_patient_flag = df_infarction_only[["PATIENT"]].drop_duplicates()
df_patient_flag["cvd_flag"] = True
df_patient_flag = df_patient_flag.reset_index(drop=True)
#df_patient_flag.to_csv("data/processed/conditions.csv", index=False)


##############################################
# List of States & fileID   (Immunizations)
##############################################

# List of States & fileID for immunizations
file_ids = {
    "PA": "1EJy_BvyLxqnyfa4XNzXPpByXP2u0E2kN",
    "IL": "1375IiYSHpT0Qc-EG4pWFIBP_x43hvEEp",
    "FL": "133gl7hPScHwYUL8YDqBK50f6l5JYFBbI",
    "TX": "16-F55tPQ2v9rLe6L4wwstmfqaY2fDck7",
    "CA": "1vshSteqa-0RLRjgo4eS-GMHRbeqTs_Vp"
}
# List to store each state's DataFrame
df_list = []
# Loop over each state and file_id
for state, fid in file_ids.items():
    df_state = load_csv_from_gdrive(fid)
    df_list.append(df_state)
# Combine all DataFrames into one
df_combined = pd.concat(df_list, ignore_index=True)

# Keep only the PATIENT and DESCRIPTION columns and drop duplicate rows
df_immunizations = df_combined[['PATIENT', 'DESCRIPTION']]
df_immunizations_unique = df_immunizations.drop_duplicates()
#df_immunizations_unique.to_csv("data/processed/immunizations.csv", index=False)

##############################################
# List of States & fileID   (Observations)
##############################################
# List of States & fileID for Observations
file_ids = {
    "PA": "1-S5akf1qovDo3jPMhAxVxnlZBbyY0oy_",
    "IL": "1sCpZzIZ6vPYo3lGO2Tglk0oyOt0sAmd3",
    "FL": "1Kk84FxRkbphPbDbte1yTEJAFr_DkgpQp",
    "TX": "1KETnD5BOIg38zk8i6rSiN03TD8_KaW3O",
    "CA": "13yJSIaRvUDCLNPJhYdHL5hBY8watl054"
}
# List to store each state's DataFrame
df_list = []
# Loop over each state and file_id
for state, fid in file_ids.items():
    df_state = load_csv_from_gdrive(fid)
    df_list.append(df_state)
# Combine all DataFrames into one
df_combined = pd.concat(df_list, ignore_index=True)
df_combined["DATE"] = pd.to_datetime(df_combined["DATE"], errors="coerce")
# Sort DATE so that the latest record comes last for each group
df_combined.sort_values("DATE", inplace=True)
# Group by PATIENT and DESCRIPTION, then keep only the last record in each group
df_latest = df_combined.groupby(["PATIENT", "DESCRIPTION"], as_index=False).tail(1)
#df_latest.to_csv("data/processed/observations.csv", index=False)

##############################################
# List of States & fileID   (Medications)
##############################################
# List of States & fileID for medications
file_ids = {
    "PA": "1nZg078K-AmEIkP0g7FQRbYvgrgfavyIp",
    "IL": "1nvqHDEClnz8ktLWSLCiPgQeHZH-BkzuT",
    "FL": "1JvcRPYyT5CddZGyqdkqy9GwvH8TkBKMJ",
    "TX": "1CF34FZMYgq9_ZQH0v9gAGlnqXrm7dk59",
    "CA": "1u0i-JV53eKeCVNTPuWxtiI4fLHCOFoAN"
}

# List to store each state's DataFrame
df_list = []
# Loop over each state and file_id
for state, fid in file_ids.items():
    df_state = load_csv_from_gdrive(fid)
    df_list.append(df_state)
# Combine all DataFrames into one
df_combined = pd.concat(df_list, ignore_index=True)

# Keep only the PATIENT and DESCRIPTION columns and drop duplicate rows
df_medications = df_combined[['PATIENT', 'DESCRIPTION']]
df_medications_unique = df_medications.drop_duplicates()

#df_medications_unique.to_csv("data/processed/medications.csv", index=False)


##############################################
# List of States & fileID   (Allergies)
##############################################
# List of States & fileID for medications
file_ids = {
    "PA": "1GAIiriodO00hYDDezmoLnbu-pQNkwdqH",
    "IL": "1ZmsXQH0AxtPQ6fOCf2bUsadgDoSyfjOw",
    "FL": "1jHxoCcxR9q48aWZ5SCxLlQAarmBiauKi",
    "TX": "17ric2bfSiyxhdy2tgeLfxySBSLI_ubPN",
    "CA": "1tCqSQAqPgRhCLv-UFWSzWfJtOpwxFMcB"
}

# List to store each state's DataFrame
df_list = []
# Loop over each state and file_id
for state, fid in file_ids.items():
    df_state = load_csv_from_gdrive(fid)
    df_list.append(df_state)
# Combine all DataFrames into one
df_combined = pd.concat(df_list, ignore_index=True)

# Keep only the PATIENT and DESCRIPTION columns and drop duplicate rows Filter where TYPE equals "allergy"
# Clean up actual allergies too
df_allergy_filtered = df_combined[df_combined["TYPE"].str.lower() == "allergy"]
df_allergy_filtered = df_allergy_filtered[~df_allergy_filtered["DESCRIPTION"].str.contains(r"Allergy to substance \(finding\)", case=False, na=False)]
df_allergy_filtered["DESCRIPTION"] = df_allergy_filtered["DESCRIPTION"].str.replace(r"\s*\(.*", "", regex=True)
df_allergies = df_allergy_filtered[['PATIENT', 'DESCRIPTION']]
df_allergies_unique = df_allergies.drop_duplicates()
#df_allergies_unique.to_csv("data/processed/allergies.csv", index=False)

###################
#  Patients all   #
###################

medications = df_medications_unique #pd.read_csv(r"data/processed/medications.csv")
conditions = df_patient_flag #pd.read_csv(r"data/processed/conditions.csv")
immunizations = df_immunizations_unique #df_immunizations_unique #pd.read_csv(r"data/processed/immunizations.csv")
observations = df_latest #pd.read_csv(r"data/processed/observations.csv")
allergies = df_allergies_unique #pd.read_csv(r"data/processed/allergies.csv")

# Rename to match the others
patients_all.rename(columns={"Id": "PATIENT"}, inplace=True)

###################
#  Observations   #
###################
#clean what will become variables
observations["clean_desc"] = observations["DESCRIPTION"].astype(str).str.strip().str.lower().str.replace(r'\W+', '_',
                                                                                                         regex=True)
f_wide_observations = observations.pivot_table(
    index="PATIENT",
    columns="clean_desc",
    values="VALUE",
    aggfunc='first'  # only 1 already but keeping it here
).reset_index()

#they need to be numeric
for col in f_wide_observations.columns:
    if col != "PATIENT":
        f_wide_observations[col] = pd.to_numeric(f_wide_observations[col], errors='coerce')

###################
#  Allergies      #
###################

allergies["clean_desc"] = allergies["DESCRIPTION"].astype(str).str.strip().str.lower().str.replace(r'\W+', '_',
                                                                                                   regex=True)
allergies["has_allergy"] = 1
df_allergies_wide = allergies.pivot_table(
    index="PATIENT",
    columns="DESCRIPTION",
    values="has_allergy",
    aggfunc="max",  # 1 remains 1
    fill_value=0  # Fill missing values with 0
).reset_index()

###################
#  medications    #
###################
medications["clean_desc"] = medications["DESCRIPTION"].astype(str).str.strip().str.lower().str.replace(r'\W+', '_',
                                                                                                           regex=True)
medications["has_med"] = 1
df_medications_wide = medications.pivot_table(
    index="PATIENT",
    columns="DESCRIPTION",
    values="has_med",
    aggfunc="max",  # 1 remains 1
    fill_value=0  # Fill missing values with 0
).reset_index()

###################
#  immunizations  #
###################
immunizations["clean_desc"] = immunizations["DESCRIPTION"].astype(str).str.strip().str.lower().str.replace(r'\W+', '_',
                                                                                                           regex=True)
immunizations["has_imm"] = 1
df_immunizations_wide = immunizations.pivot_table(
    index="PATIENT",
    columns="DESCRIPTION",
    values="has_imm",
    aggfunc="max",  # 1 remains 1
    fill_value=0  # Fill missing values with 0
).reset_index()

merged_df = patients_all.copy()
# Merge the observations dataset
merged_df = merged_df.merge(f_wide_observations, on="PATIENT", how="left")
# Merge the allergies dataset
merged_df = merged_df.merge(df_allergies_wide, on="PATIENT", how="left")
# Merge the medications dataset
merged_df = merged_df.merge(df_medications_wide, on="PATIENT", how="left")
# Merge the immunizations dataset
merged_df = merged_df.merge(df_immunizations_wide, on="PATIENT", how="left")
# Merge the conditions dataset
merged_df = merged_df.merge(conditions, on="PATIENT", how="left")

obs_cols = f_wide_observations.columns.tolist()
cols_to_fill = [col for col in merged_df.columns if col not in obs_cols]
merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(0)

#patients with no record in the conditions dataset, fill cvd_flag with False.
merged_df["cvd_flag"] = merged_df["cvd_flag"].fillna(False).astype(int)

#split 80% test, 20% train.
shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * len(shuffled_df))
train_df = shuffled_df.iloc[:train_size]
test_df = shuffled_df.iloc[train_size:]

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)


