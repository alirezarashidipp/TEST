import pandas as pd

def clean_df(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def load_mapping(mapping_path, mapping_cols, tech_col):
    mapping = pd.read_excel(mapping_path)
    mapping = clean_df(mapping, mapping_cols + [tech_col])
    mapping["SPEC"] = mapping[mapping_cols].ne("*").sum(axis=1)
    return mapping.sort_values("SPEC", ascending=False)

def classify_row(row, mapping, mapping_cols, employee_cols, tech_col):
    for _, rule in mapping.iterrows():
        if all(
            rule[m] == "*" or rule[m] == row[e]
            for m, e in zip(mapping_cols, employee_cols)
        ):
            return rule[tech_col]
    return "unknown"

def apply_mapping(
        df,
        mapping_path,
        mapping_cols,
        employee_cols,
        tech_col="TECH_FLAG",
        new_col="TECH_FLAG"
    ):

    df = clean_df(df, employee_cols)
    mapping = load_mapping(mapping_path, mapping_cols, tech_col)

    df[new_col] = df.apply(
        lambda row: classify_row(row, mapping, mapping_cols, employee_cols, tech_col),
        axis=1
    )
    return df






import pandas as pd
from role_mapper import apply_mapping

employees = pd.read_excel("employees.xlsx")

df_tagged = apply_mapping(
    employees,
    mapping_path="mapping.xlsx",
    mapping_cols=["JOB_FAMILY", "JOB_SUB_FAMILY", "JOB_CATEGORY"],  
    employee_cols=["FAMILY_NAME", "SUB_FAM", "CATEGORY_NAME"],       
    tech_col="TECH_FLAG",                                           
    new_col="TECH_FLAG"                                              
)

df_tagged.to_excel("tagged.xlsx", index=False)
