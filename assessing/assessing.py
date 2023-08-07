from sklearn.model_selection import cross_validate, LeaveOneGroupOut
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# assuming df is your dataframe, X are your features, y is your target
df = pd.read_csv('descriptors_lin_nonlin_etc.csv')  # replace with your data file

df = df.loc[df['graph_id']<500]
X = df.drop(columns=['graph_id','edge_source','edge_dest', 'is_causal'])
y = df['is_causal']

# Leave-One-Group-Out cross validator
logo = LeaveOneGroupOut()

# create a Logistic Regression classifier
classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# use cross_validate and fit it with LOGO cross-validator
scores = cross_validate(classifier, X, y, cv=logo.split(X, y, df['graph_id']), n_jobs=-1, 
                        scoring=['accuracy', 'f1', 'roc_auc'], return_train_score=True)

print("Test Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
print("Test F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))
print("Test ROC AUC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))
