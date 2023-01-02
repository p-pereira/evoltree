# **evoltree**: Evolutionary Decision Trees

[![Downloads](https://pepy.tech/badge/evoltree)](https://pepy.tech/project/evoltree) [![Downloads](https://pepy.tech/badge/evoltree/month)](https://pepy.tech/project/evoltree) [![Downloads](https://pepy.tech/badge/evoltree/week)](https://pepy.tech/project/evoltree)


## Overview

*evoltree* is a novel **Multi-objective Optimization (MO)** approach to evolve **Decision Trees (DT)** using **Grammatical Evolution (GE)**, under two main variants: a pure GE method (**EDT**) and a GE with Lamarckian Evolution (**EDTL**).
Both variants evolve variable-length DTs and perform a simultaneous optimization of the predictive performance (measured in terms of AUC) and model complexity (measured in terms of GE tree nodes). To handle big data, the GE methods include a **training sampling** and **parallelism evaluation mechanism**.
Both variants both use [PonyGE2](https://github.com/PonyGE/PonyGE2) as GE engine, while EDTL uses [sklearn DT](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).


Solutions are represented as a **numpy.where** expression in the format bellow, with *x* being a pandas dataframe with data; *idx* a column from the dataset; *comparison* a comparison signal (e.g., <, ==); *value* being a numerical value; and *result* can be another **numpy.where** expression (creating chained expressions), or a class probability (numeric value from 0 to 1).
Due to this representation, current evoltree implementation only allows numerical attributes.

```python3
numpy.where(x[<idx>]<comparison><value>,<result>,<result>) 
```
Example of a generated Python code (left) and the corresponding DT (right):
![DT and code](https://raw.githubusercontent.com/p-pereira/evoltree/main/imgs/dt_code.png)

More details about this work can be found at: https://doi.org/10.1016/j.eswa.2020.114287.

# Install

**Using `pip`:**

```bash
pip install evoltree
```

# Quick Start

This short tutorial contains a set of steps that will help you getting started with **evoltree**.

## Load Example Data

evoltree package includes two example datasets for testing purposes only. Data is ordered in time and the second dataset contains events collected after the first one.
To load the datasets, already divided into train, validation and test sets, two functions were created:
- **load_offline_data** - returns the training, validation and test sets from first dataset, used for static environments;
- **load_online_data** - returns the training, validation and test sets from both datasets, used for online learning scenarios.

Next steps present how to load data in the two different modes (online and offline). Due to privacy issues, all data is anonymized.

```python3
# Import package
from evoltree import evoltree
# Create evoltree object
edt = evoltree()
# Loading first example dataset, already divided into train, validation and test sets.
X, y, X_val, y_val, X_ts, y_ts = edt.load_offline_data()
# Loading both example datasets, already divided into train, validation and test sets.
X1, y1, X_val1, y_val1, X_ts1, y_ts1, X2, y2, X_val2, y_val2, X_ts2, y_ts2 = edt.load_online_data()
print(X)
print(y)
```

```
# Outputs
## X (train)
        col1      col2      col3      col4      col5      col6      col7      col8      col9      col10
0       0.755951  4.653432  2.767041  0.739897  0.101081  2.401580  2.890712  3.157321  5.554321  0.865465
1       5.624979  4.782823  0.416159  0.823179  0.101081  2.446735  0.739841  2.564807  4.552277  0.991334
2       0.755951  5.365407  3.081596  0.739897  0.101081  3.174113  1.986985  3.193263  3.378832  0.865465
3       4.436114  6.393779  0.416159  0.823179  0.101081  2.238128  2.066007  2.564807  3.436435  0.865465
4       7.071106  6.069200  0.416159  0.739897  0.101081  5.100103  0.739841  3.551982  4.886248  0.991334
...          ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
708937  0.755951  4.804125  0.416159  0.823179  0.101081  2.807017  2.066007  2.564807  5.802875  0.865465
708938  0.755951  2.558737  0.416159  0.823179  0.101081  2.558737  2.066007  2.564807  5.802875  2.173501
708939  0.755951  2.875355  0.416159  0.823179  0.101081  2.572643  0.739841  2.455024  2.947305  0.991334
708940  0.755951  4.694221  0.416159  0.823179  0.101081  3.568007  0.739841  2.455024  6.279122  0.991334
708941  6.631839  2.553551  0.416159  0.823179  0.101081  2.446735  0.739841  2.564807  5.266229  0.991334

[708942 rows x 10 columns]

[708942 rows x 10 columns]

## y (train)
0         NoSale
1         NoSale
2         NoSale
3         NoSale
4         NoSale
           ...
708937    NoSale
708938    NoSale
708939    NoSale
708940    NoSale
708941    NoSale
Name: target, Length: 708942, dtype: object
```

## Offline Learning: Fit EDT and EDTL models

Next steps present the basic usage of both variants (EDT and EDTL) for modeling the previously loaded data in an offline environment. Furthermore, since all solutions are stored, it is possible to continue the learning process if needed, by using the **refit** function, also presented below.

```python3
# Imports
from evoltree import evoltree
from sklearn import metrics
import matplotlib.pyplot as plt
# Create two evoltree objects, one for each variant
edt = evoltree()
edtl = evoltree()
# Load dataset
X, y, X_val, y_val, X_ts, y_ts = edt.load_offline_data()
# Fit both versions on train data
## Normal variant:
edt.fit(X, y, "Sale", X_val, y_val, pop=100, gen=10, lamarck=False, experiment_name="test")
## Lamarckian variant, doesn't need as much iterations (gen)
edtl.fit(X, y, "Sale", X_val, y_val, pop=100, gen=5, lamarck=True, experiment_name="testLamarck")
# Continue Fitting both versions on the same datasets for extra 2 iterations
## Normal variant:
edt.refit(gen=2)
## Lamarckian variant, doesn't need as much iterations (gen)
edtl.refit(gen=2)
# Predict on test data, using the solution with better predictive performance on validation data
y_pred1 = edt.predict(X_ts, mode="best")
y_predL1 = edtl.predict(X_ts, mode="best")
# Compute AUC on test data
fpr1, tpr1, th1 = metrics.roc_curve(y_ts, y_pred1, pos_label='Sale')
fprL1, tprL1, thL1 = metrics.roc_curve(y_ts, y_predL1, pos_label='Sale')
auc1 = metrics.auc(fpr1, tpr1)
aucL1 = metrics.auc(fprL1, tprL1)
# Plot results
fig, ax = plt.subplots(1,1, figsize=(5.5,5))
plt.plot(fprL1, tprL1, color='royalblue', ls="--", lw=2, label="EDTL={}%".format(round(aucL1, 2)))
plt.plot(fpr1, tpr1, color='darkorange', ls="-", lw=2, label="EDT={}%".format(round(auc1, 2)))
plt.plot([0,1], [0,1], color="black", ls='--', label="baseline=50%")
plt.legend(loc=4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("results.png")

```
Result:

![Results.](https://raw.githubusercontent.com/p-pereira/evoltree/main/imgs/results.png)


## Online Learning: Fit EDT and EDTL models

EDT variants can both be applied to online learning environmnets, saving previous solutions and using it as starting point for the learning process, thus, needing a smaller number of iterations to achieve good results.
Next steps show how to implement it.

```python3
# Imports
from evoltree import evoltree
from sklearn import metrics
import matplotlib.pyplot as plt
# Create two evoltree objects, one for each variant
edt = evoltree()
edtl = evoltree()
# Load datasets
X1, y1, X_val1, y_val1, X_ts1, y_ts1, X2, y2, X_val2, y_val2, X_ts2, y_ts2 = edt.load_online_data()
### Train models: first dataset
# Fit both versions on train data
## Normal variant:
edt.fit(X1, y1, "Sale", X_val1, y_val1, pop=100, gen=10, lamarck=False, experiment_name="test")
## Lamarckian variant, doesn't need as much iterations (gen)
edtl.fit(X1, y1, "Sale", X_val1, y_val1, pop=100, gen=5, lamarck=True, experiment_name="testLamarck")
# Predict on test data, using the solution with better predictive performance on validation data
y_pred1 = edt.predict(X_ts1, mode="best")
y_predL1 = edtl.predict(X_ts1, mode="best")
# Compute AUC on test data
fpr1, tpr1, th1 = metrics.roc_curve(y_ts1, y_pred1, pos_label='Sale')
fprL1, tprL1, thL1 = metrics.roc_curve(y_ts1, y_predL1, pos_label='Sale')
auc1 = metrics.auc(fpr1, tpr1)
aucL1 = metrics.auc(fprL1, tprL1)
### Re-Train models: second dataset
# Fit both versions on train data
## Normal variant:
edt.fit_new_data(X2, y2, X_val2, y_val2, pop=100, gen=5, lamarck=False)
## Lamarckian variant, doesn't need as much iterations (gen)
edtl.fit_new_data(X2, y2, X_val2, y_val2, pop=100, gen=2, lamarck=True)
# Predict on test data, using the solution with better predictive performance on validation data
y_pred2 = edt.predict(X_ts2, mode="best")
y_predL2 = edtl.predict(X_ts2, mode="best")
# Compute AUC on test data
fpr2, tpr2, th2 = metrics.roc_curve(y_ts2, y_pred2, pos_label='Sale')
fprL2, tprL2, thL2 = metrics.roc_curve(y_ts2, y_predL2, pos_label='Sale')
auc2 = metrics.auc(fpr2, tpr2)
aucL2 = metrics.auc(fprL2, tprL2)
# Plot results
fig, ax = plt.subplots(1,1, figsize=(5.5,5))
plt.plot(fprL1, tprL1, color='royalblue', ls="--", lw=2, label="EDTL (1)={}%".format(round(aucL1, 2)))
plt.plot(fpr1, tpr1, color='darkorange', ls="-", lw=2, label="EDT (1)={}%".format(round(auc1, 2)))
plt.plot(fprL2, tprL2, color='navy', ls="--", lw=2, label="EDTL (2)={}%".format(round(aucL2, 2)))
plt.plot(fpr2, tpr2, color='tan', ls="-", lw=2, label="EDT (2)={}%".format(round(auc2, 2)))
plt.plot([0,1], [0,1], color="black", ls='--', label="baseline=50%")
plt.legend(loc=4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("results_online.png")

```

Results:

![results_online](https://raw.githubusercontent.com/p-pereira/evoltree/main/imgs/results_online.png)

# Citation

If you use **evoltree** for your research, please cite the following paper:


Pedro Jose Pereira, Paulo Cortez, Rui Mendes:

[**Multi-objective Grammatical Evolution of Decision Trees for Mobile Marketing User Conversion Prediction.**](https://doi.org/10.1016/j.eswa.2020.114287)

Expert Systems with Applications 168: 114287 (2021).


```
@article{DBLP:journals/eswa/PereiraCM21,
	author    = {Pedro Jos{\'{e}} Pereira and Paulo Cortez and Rui Mendes},
	title     = {Multi-objective Grammatical Evolution of Decision Trees for Mobile Marketing user conversion prediction},
	journal   = {Expert Systems with Applications},
	volume    = {168},
	pages     = {114287},
	year      = {2021},
	doi       = {10.1016/j.eswa.2020.114287},
}
```
