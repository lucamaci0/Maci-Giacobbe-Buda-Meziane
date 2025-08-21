# Student Performance Analysis Project

**Dataset:** [Kaggle – Student Performance Data Set](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
**Objective:** Analyze student data to predict academic performance and identify groups with similar characteristics.

## Group Tasks

### 1. Preprocessing and Exploratory Analysis

* Clean the dataset, normalize numerical variables, and encode categorical variables.
* Explore data distributions and identify key correlations.
  **Contributor:** Alessio Buda

### 2. Supervised Learning

* Train a classification model (Decision Tree) to predict whether a student will have “high” or “low” performance based on socio-demographic and school-related variables. I considered all features (except StudentID and TargetClass) for training the model. 
* Evaluate the model using a classification report and confusion matrix.
* As an EXTRA: use of Cross-validation fold and its report. implementation of a Grid search for finding the best parameters.
*
  **Contributor:** Beatrice Giacobbe

### 3. Unsupervised Learning

* Apply clustering algorithms to identify groups of students with similar characteristics:

  * **K-Means:** Luca Maci
  * **DBSCAN:** Cherki Meziane
* Compare clusters with supervised labels to identify consistent patterns.

### 4. Extra

* Implement a SciPy workflow to allow statistical queries on the input data.


