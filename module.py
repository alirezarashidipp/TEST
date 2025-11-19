import pandas as pd

# default mapping path inside the module
MAPPING_PATH = "C:/Projects/MappingFiles/role_mapping.xlsx"

def clean_df(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def load_mapping(mapping_cols, tech_col):
    mapping = pd.read_excel(MAPPING_PATH)
    mapping = clean_df(mapping, mapping_cols + [tech_col])
    mapping["SPEC"] = mapping[mapping_cols].ne("*").sum(axis=1)
    return mapping.sort_values("SPEC", ascending=False)

def mapper(mapping_cols, employee_cols, tech_col):
    mapping = load_mapping(mapping_cols, tech_col)

    def classify(row):
        for _, rule in mapping.iterrows():
            if all(rule[m] == "*" or rule[m] == row[e]
                   for m, e in zip(mapping_cols, employee_cols)):
                return rule[tech_col]
        return "unknown"

    return classify





import pandas as pd
from role_mapper import mapper

employees = pd.read_excel("employees.xlsx")

# names from employee df
employee_cols = ["FAMILY_NAME", "SUB_FAM", "CATEGORY_NAME"]

# names from mapping file
mapping_cols = ["JOB_FAMILY", "JOB_SUB_FAMILY", "JOB_CATEGORY"]
tech_col     = "TECH_FLAG"

output_col = "ROLE_TYPE"

# clean df
for c in employee_cols:
    employees[c] = employees[c].astype(str).str.strip().str.lower()

# build classifier — mapping file is read INSIDE the module
classifier = mapper(mapping_cols, employee_cols, tech_col)

# apply mapping
employees[output_col] = employees.apply(classifier, axis=1)

employees.to_excel("tagged.xlsx", index=False)
