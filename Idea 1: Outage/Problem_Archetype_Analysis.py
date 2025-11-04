import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Data preprocessing
numerical_features = ['OUTAGE_DURATION_MINUTES_NUM']
categorical_features = [
    'CMDB_CONFIGURATION_TTEM_NAME',
    'OUTAGE_TYPE_NAME',
    'IT_INCIDENT_SEVERITY_NAME',
    'PROBLEM_TASK_CAUSE_CODE',
    'CHANGE_ASSIGNMENT_GROUP_NAME'
]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

X_processed = preprocessor.fit_transform(df)

# Elbow method to find optimal k
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Apply K-means clustering with chosen k
CHOSEN_K = 3
kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_processed)
df['cluster'] = cluster_labels

# Visualize clusters with PCA
pca = PCA(n_components=2)
X_pca = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
X_pca = pca.fit_transform(X_pca)

plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab10', alpha=0.7)
plt.title(f'Outage Clusters (k={CHOSEN_K}) Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Analyze cluster characteristics
for i in range(CHOSEN_K):
    cluster_df = df[df['cluster'] == i]
    print(f"CLUSTER {i} SIZE: {len(cluster_df)}")
    print(cluster_df[numerical_features].describe().loc[['mean', 'std', 'min', 'max']])
    for col in categorical_features:
        print(f"\n{col} top categories:")
        print(cluster_df[col].value_counts(normalize=True).nlargest(3) * 100)
    print('-' * 50)
