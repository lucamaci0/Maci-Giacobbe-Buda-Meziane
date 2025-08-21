import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

path = "Student_performance_data _.csv"
df = pd.read_csv(path)

print("The shape of the datset is:  ", df.shape)
 
# check for null values
print(df.isnull().sum())
 
# check for duplicate rows
print(df.duplicated().sum())
  
# clean the dataset
df = df.dropna()  # remove rows with missing values
df = df.drop_duplicates()  # remove duplicate rows
df = df.reset_index(drop=True)  # reset the index after dropping rows
 
# Display the cleaned DataFrame
print(df.head())
#le = LabelEncoder()
#df["Genre"] = le.fit_transform(df["GradeClass"])

df["GradeClass"] = df["GradeClass"].astype(int)
print(type((df.loc[0,'GradeClass'])))
print(df.head())

# Remove outliers using Z-score method
categorical_cols = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring',
                    'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering' ]
 
 
 
for col in categorical_cols:
    df[col] = df[col].astype('category')
 
print(df.dtypes)

x = df.drop(columns=['StudentID','GradeClass'])
y = [0 if el >= 3 else 1 for el in df['GradeClass']]

# PUNTO 2 - DECISION TREE
# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.25, random_state=42
)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
score = cross_val_score(tree, x, y, cv=skf, scoring="roc_auc")

plt.figure(figsize=(18, 10))
plot_tree(tree, feature_names=x.columns, class_names=["Basso","Alto"], filled=True)
plt.title("Albero Decisionale")
plt.show()

print("Decision Tree:")
print(classification_report(y_test, y_pred, digits=3))

print("Score di CV:\nMedia:",score.mean(),"\nStd:", score.std())

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


