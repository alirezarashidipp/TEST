import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# STEP 0: SIMULATE YOUR DATA
# (You will replace this section with: df = pd.read_csv('your_file.csv'))
# ------------------------------------------------------------------
print("Step 0: Simulating data...")
data = {
    'OUTAGE_ID': [f'OUT{1000 + i}' for i in range(500)],
    'CMDB_CONFIGURATION_ITEM_NAME': np.random.choice(['DB-Server-01', 'App-Server-Prod', 'Network-Switch-A', 'Auth-Service', 'Payment-Gateway'], 500),
    'OUTAGE_TYPE_NAME': np.random.choice(['Degradation', 'Outage', 'Planned Outage'], 500, p=[0.4, 0.5, 0.1]),
    'OUTAGE_DURATION_MINUTES_NUM': np.abs(np.random.normal(120, 80, 500)).astype(int),
    'IT_INCIDENT_SEVERITY_NAME': np.random.choice(['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4'], 500, p=[0.1, 0.3, 0.4, 0.2]),
    'PROBLEM_TASK_CAUSE_CODE': np.random.choice(['Software-Bug', 'Hardware-Failure', 'Change-Related', 'Vendor-Issue', 'Unknown'], 500),
    'CHANGE_ASSIGNMENT_GROUP_NAME': np.random.choice(['Infra-Team', 'App-Dev-Team', 'Network-Ops', 'DB-Admins'], 500)
}
df = pd.DataFrame(data)

# --- Simulate some patterns for the clusters to find ---
# Cluster 1: 'Vendor-Issue' causing long 'Sev 1' outages
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Vendor-Issue', 'OUTAGE_DURATION_MINUTES_NUM'] = np.random.normal(300, 50, df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Vendor-Issue'].shape[0]).astype(int)
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Vendor-Issue', 'IT_INCIDENT_SEVERITY_NAME'] = 'Sev 1'
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Vendor-Issue', 'CMDB_CONFIGURATION_ITEM_NAME'] = 'Payment-Gateway'

# Cluster 2: 'Change-Related' degradations from 'App-Dev-Team'
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Change-Related', 'OUTAGE_TYPE_NAME'] = 'Degradation'
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Change-Related', 'CHANGE_ASSIGNMENT_GROUP_NAME'] = 'App-Dev-Team'
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Change-Related', 'IT_INCIDENT_SEVERITY_NAME'] = 'Sev 3'
df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Change-Related', 'OUTAGE_DURATION_MINUTES_NUM'] = np.random.normal(45, 10, df.loc[df['PROBLEM_TASK_CAUSE_CODE'] == 'Change-Related'].shape[0]).astype(int)

print("Data simulation complete. Showing sample:")
print(df.head())
print("-" * 60)

# ------------------------------------------------------------------
# STEP 1: PREPROCESSING
# Define which columns are numerical and which are categorical
# ------------------------------------------------------------------
print("Step 1: Preprocessing data...")

# We'll use duration as our only numerical feature for clustering
numerical_features = ['OUTAGE_DURATION_MINUTES_NUM']

# All other features are categories
categorical_features = [
    'CMDB_CONFIGURATION_ITEM_NAME',
    'OUTAGE_TYPE_NAME',
    'IT_INCIDENT_SEVERITY_NAME',
    'PROBLEM_TASK_CAUSE_CODE',
    'CHANGE_ASSIGNMENT_GROUP_NAME'
]

# Create a preprocessing pipeline
# This handles all steps automatically:
# 1. Impute (fill in) missing values
# 2. Scale numerical data (so 'duration' doesn't dominate)
# 3. One-hot encode categorical data (convert text to numbers)

# Pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Fill missing durations with the median
    ('scaler', StandardScaler())                    # Scale to standard normal distribution
])

# Pipeline for categorical features
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), # Fill missing text with 'Unknown'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # Convert categories to 1s and 0s
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

# Apply the preprocessing
X_processed = preprocessor.fit_transform(df)

print("Preprocessing complete. Data shape:", X_processed.shape)
print("-" * 60)

# ------------------------------------------------------------------
# STEP 2: FIND OPTIMAL 'k' (THE ELBOW METHOD)
# We plot the 'inertia' (sum of squared distances) for different k values.
# The "elbow" of the curve is the best trade-off.
# ------------------------------------------------------------------
print("Step 2: Finding optimal 'k' using the Elbow Method...")
inertia = []
K_range = range(1, 11) # Test k from 1 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
print("Showing Elbow Plot. Close the plot window to continue...")
plt.show()

print("-" * 60)
# --- USER INTERACTION ---
# In a real scenario, you'd look at the graph and pick the 'k'
# where the line starts to bend (the "elbow").
# Let's assume the elbow is at k=4 for this example.
CHOSEN_K = 4
print(f"Step 3: Running K-Means with chosen k = {CHOSEN_K}...")
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# STEP 3: RUN K-MEANS CLUSTERING
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_processed)

# Add the cluster labels back to our original DataFrame
df['cluster'] = cluster_labels
print("Clustering complete. Cluster labels added to DataFrame.")
print(df.head())
print("-" * 60)

# ------------------------------------------------------------------
# STEP 4: VISUALIZE THE CLUSTERS (using PCA)
# We need to reduce the many processed dimensions (e.g., 50+)
# down to 2 dimensions (x, y) to plot them.
# ------------------------------------------------------------------
print("Step 4: Visualizing clusters with PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed.toarray()) # Convert sparse matrix to dense

# Create a DataFrame for plotting
pca_df = pd.DataFrame(data=X_pca, columns=['PCA_1', 'PCA_2'])
pca_df['cluster'] = cluster_labels

# Plot the 2D PCA
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PCA_1', y='PCA_2',
    hue='cluster',
    palette=sns.color_palette('hsv', n_colors=CHOSEN_K),
    data=pca_df,
    legend='full',
    alpha=0.7
)
plt.title(f'Outage Clusters (k={CHOSEN_K}) Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
print("Showing PCA Cluster Plot. Close the plot window to continue...")
plt.show()

print("-" * 60)

# ------------------------------------------------------------------
# STEP 5: ANALYZE AND PROFILE THE CLUSTERS
# This is the most important part for your managers.
# What does each cluster *mean*?
# ------------------------------------------------------------------
print("Step 5: Profiling cluster characteristics...")

for i in range(CHOSEN_K):
    cluster_df = df[df['cluster'] == i]
    print(f"\n======= CLUSTER {i} PROFILE (Size: {len(cluster_df)} outages) =======")
    
    # Analyze numerical features
    print("\n--- Numerical Stats ---")
    print(cluster_df[numerical_features].describe().loc[['mean', 'std', 'min', 'max']])
    
    # Analyze categorical features (show top 3 most common)
    print("\n--- Top Categorical Features ---")
    for col in categorical_features:
        print(f"  {col}:")
        print(cluster_df[col].value_counts(normalize=True).nlargest(3) * 100)
        print("-" * 20)

print("\nAnalysis complete. Review the cluster profiles to name your 'archetypes'.")
