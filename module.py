import pandas as pd

# Hardcoded configs inside the module
MAPPING_PATH = "C:/Projects/MappingFiles/role_mapping.xlsx"
TECH_COL = "TECH_FLAG"

def clean_df(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def load_mapping(mapping_cols):
    mapping = pd.read_excel(MAPPING_PATH)
    mapping = clean_df(mapping, mapping_cols + [TECH_COL])
    mapping["SPEC"] = mapping[mapping_cols].ne("*").sum(axis=1)
    return mapping.sort_values("SPEC", ascending=False)

def mapper(mapping_cols, employee_cols):
    mapping = load_mapping(mapping_cols)

    def classify(row):
        for _, rule in mapping.iterrows():
            if all(
                rule[m] == "*" or rule[m] == row[e]
                for m, e in zip(mapping_cols, employee_cols)
            ):
                return rule[TECH_COL]
        return "unknown"

    return classify













import pandas as pd
from role_mapper import mapper

employees = pd.read_excel("employees.xlsx")

mapping_cols  = ["JOB_FAMILY", "JOB_SUB_FAMILY", "JOB_CATEGORY"]
employee_cols = ["FAMILY_NAME", "SUB_FAM", "CATEGORY_NAME"]

output_col = "ROLE_TYPE"

# clean original df
for c in employee_cols:
    employees[c] = employees[c].astype(str).str.strip().str.lower()

# get classifier (mapping + TECH_COL + mapping path loaded from module)
classifier = mapper(mapping_cols, employee_cols)

employees[output_col] = employees.apply(classifier, axis=1)

employees.to_excel("tagged.xlsx", index=False)
