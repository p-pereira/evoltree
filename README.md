# **MGEDT**: Multiobjective Grammatical Evolution Decision Trees for classification tasks

## Overview

MGEDT is a novel **Multiobjective Optimization (MO)** approach to evolve **Decision Trees (DT)** using **Grammatical Evolution (GE)**, under two main variants: a pure GE method (**MGEDT**) and a GE with Lamarckian Evolution (**MGEDTL**).
Both variants evolve variable-length DTs and perform a simultaneous optimization of the predictive performance (measured in terms of AUC) and model complexity (measured in terms of GE tree nodes). To handle big data, the GE methods include a **training sampling** and **parallelism evaluation mechanism**.
Both variants both use [PonyGE2](https://github.com/PonyGE/PonyGE2) as GE engine, while MGEDTL uses [sklearn DT](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).


Solutions are represented as a **numpy.where** expression in the fromat bellow, with *x* being a pandas dataframe with data; *idx* a column from the dataset; *comparison* a comparison signal (e.g., <, ==); *value* being a numerical value; and *result* can be another **numpy.where** expression (creating chained expressions), or a class probabability (numeric value from 0 to 1).
Due to this representation, current MGEDT implementation only allows numerical attributes.

```python3
numpy.where(x[<idx>]<comparison><value>,<result>,<result>) 
```
Example of a generated Python code (left) and the corresponding DT (right):
![Example of a generated Python code (left) and the corresponding DT (right).](https://github.com/p-pereira/MGEDT/blob/dev/imgs/dt_code.png)

More detais about this work can be found at: https://doi.org/10.1016/j.eswa.2020.114287.

# Install

**Using `pip`:**

```bash
pip install MGEDT
```

# Quick Start

This short tutorial contains a set of steps that will help you getting started with **MGEDT**.

## Load Example Data

MGEDT package includes an example dataset for testing purposes only. Next steps show how to load it.

```python3
# Import package
from MGEDT import MGEDT
# Create MGEDT object
mgedt = MGEDT()
# Loading example dataset, already divided into train, validation and test sets.
X, y, X_val, y_val, X_ts, y_ts = mgedt.load_example_data()
print(X)
print(y)
```

```
# Outputs
## X (train)
         idoperator  idcampaign  idbrowser  idverticaltype  idapplication  idpartner  idaffmanager  regioncontinent  country_name  accmanager
0         0.755951    2.875355   0.416159        0.823179       0.101081   2.572643      2.028685         2.689715      4.957863    0.865465
1         0.755951   11.379666   3.005727        3.662859      11.379666   7.131171      5.018939         3.477055      5.740521    2.173501
2         0.755951    5.365407   3.005727        0.739897       0.101081   3.174113      1.986985         3.193263      3.378832    0.865465
3         0.755951    2.558737   0.416159        0.823179       0.101081   2.558737      2.028685         2.689715      7.051861    0.865465
4         8.210914    2.781405   0.416159        0.823179       0.101081   2.401580      0.739841         1.751485      5.461369    0.991334
...            ...         ...        ...             ...            ...        ...           ...              ...           ...         ...
708937    0.755951    3.766890   3.081596        0.739897       0.101081   2.942647      0.739841         2.430474      2.913392    0.865465
708938    0.755951    2.781405   0.416159        0.823179       0.101081   2.401580      2.028685         4.108194      5.259857    0.865465
708939    3.217657    2.875355   0.416159        0.823179       0.101081   2.572643      0.739841         1.751485      2.158649    0.991334
708940    3.217657    4.180984   0.416159        0.739897       0.101081   3.174113      0.739841         1.751485      2.158649    0.991334
708941    4.997474    5.141558   2.767041        0.739897       0.101081   2.238128      0.739841         2.301093      3.448381    0.865465

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

## Fit MGEDT and MGEDTL models

Next steps present the basic usage of both variants (MGEDT and MGEDTL) for modeling the previously loaded data.

```python3
# Imports
from MGEDT import MGEDT
from sklearn import metrics
import matplotlib.pyplot as plt
# Create two MGEDT objects, one for each variant
mgedt = MGEDT()
mgedtl = MGEDT()
# Load dataset
X, y, X_val, y_val, X_ts, y_ts = mgedt.load_example_data()
# Fit both versions on train data
mgedt.fit(X, y, X_val, y_val, pop=100, gen=10, lamarck=False, experiment_name="test") # Normal variant
mgedtl.fit(X, y, X_val, y_val, pop=100, gen=5, lamarck=True, experiment_name="testLamarck") # Lamarckian variant, doesn't need as much iterations (gen)
# Predict on test data, using the solution with better predictive performance on validation data
y_pred1 = mgedt.predict(X_ts, mode="best")
y_pred2 = mgedtl.predict(X_ts, mode="best")
# Compute AUC on test data
fpr1, tpr1, th1 = metrics.roc_curve(y_ts, y_pred1, pos_label='Sale')
fpr2, tpr2, th2 = metrics.roc_curve(y_ts, y_pred2, pos_label='Sale')
auc1 = metrics.auc(fpr1, tpr1)
auc2 = metrics.auc(fpr2, tpr2)
# Plot results
fig, ax = plt.subplots(1,1, figsize=(5.5,5))
plt.plot(fpr2, tpr2, color='royalblue', ls="--", lw=2,
         label="MGEDTL={}%".format(round(auc2, 2)))
plt.plot(fpr1, tpr1, color='darkorange', ls="-", lw=2,
         label="MGEDT={}%".format(round(auc1, 2)))
plt.plot([0,1], [0,1], color="black", ls='--', label="baseline=50%")
plt.legend(loc=4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("results.png")

```
Result:

![Results.](https://github.com/p-pereira/MGEDT/blob/dev/imgs/results.png)



