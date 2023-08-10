from sklearn.model_selection import cross_validate, LeaveOneGroupOut
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

# assuming df is your dataframe, X are your features, y is your target
df = pd.read_csv('edges_descriptors.csv')  # replace with your data file

# sampled_ids = df['graph_id'].drop_duplicates().sample(10)

# # Use the sampled_ids to select from the original df
# df = df[df['graph_id'].isin(sampled_ids)]

# Populate the graph with edges and attributes
graphs = {}
for _, row in df.iterrows():
    # Check if graph_id exists, if not create a new DiGraph
    if row['graph_id'] not in graphs:
        graphs[row['graph_id']] = nx.DiGraph()
    G = graphs[row['graph_id']]
    
    # Add an edge if is_causal is True
    if row['is_causal']: 
        G.add_edge(row['edge_source'], row['edge_dest'])


X = df.drop(columns=['graph_id','edge_source','edge_dest', 'is_causal'])
y = df['is_causal']

# Leave-One-Group-Out cross validator
logo = LeaveOneGroupOut()

# create a Logistic Regression classifier
classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# use cross_validate and fit it with LOGO cross-validator
scores = cross_validate(classifier, X, y, cv=logo.split(X, y, df['graph_id']), n_jobs=-1, 
                        scoring=['accuracy', 'f1', 'roc_auc'], return_train_score=True)

# print(scores)
print("Test Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
print("Test F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))
print("Test ROC AUC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))
