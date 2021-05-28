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
```python3
from MGEDT import MGEDT
# Loading example dataset, already divided into train, validation and test sets.
X, y, X_val, y_val, X_ts, y_ts = mgedt.load_example_data()
print(X)
```

```
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
```