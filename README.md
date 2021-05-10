# 5470
# *Linear* Regress Model

> All in 10-Fold

## classical linear regressors

| Model                                      | MAE Loss | Score   |                                                     |
| ------------------------------------------ | -------- | ------- | --------------------------------------------------- |
| OLS                                        | 21.11952 | 0.04521 |                                                     |
| Ridge                                      | 21.11952 | 0.04521 |                                                     |
| Lasso ($\alpha=1$)                         | 21.37201 | 0.04470 |                                                     |
| Lasso ($\alpha=0.5$)                       | 21.29881 | 0.04485 |                                                     |
| Lasso ($\alpha=0.1$)                       | 21.26457 | 0.04491 |                                                     |
| LassoCV                                    | 21.26442 | 0.04491 |                                                     |
| ElasticNet($\alpha=0.05$, $l1\_ratio=0.5$) | 21.26077 | 0.04492 |                                                     |
| Lars (Least Angle Regression)              | 21.26795 | 0.04491 |                                                     |
| LassoLars($\alpha=0.0005$)                 | 21.26465 | 0.04491 |                                                     |
| **Outlier-robust regressors**              |          |         |                                                     |
| HuberRegressor                             | 20.96725 | 0.04552 | Linear regression model that is robust to outliers. |
| RANSACRegressor                            | 25.50738 | 0.03773 | RANSAC (RANdom SAmple Consensus) algorithm.         |
| **GLM**                                    |          |         |                                                     |
| PoissonRegressor                           | 21.60692 | 0.04423 | GLM with Poisson distribution                       |
| TweedieRegressor                           | 23.14525 | 0.04142 | (power=0 :Normal )                                  |
|                                            | 21.60811 | 0.04423 | Compound Poisson Gamma (power=0.5)                  |
|                                            | 23.26714 | 0.04121 | $\gamma$-distribution (power=2)                     |
|                                            | 33.78030 | 0.02875 | Inverse Gaussian (power=3)                          |

## SVM

| Model     | MAE Loss         | Score            |                                                              |
| --------- | ---------------- | ---------------- | ------------------------------------------------------------ |
| LinearSVR | 21.00690         | 0.04544          | Linear Support Vector Regression.                            |
| NuSVR     | *No convergence* | *No convergence* | uses a parameter $\nu$ to control the number of support vectors |
| SVR       | 21.33680         | 0.04477          | kernel='rbf'                                                 |

## Tree

| Model                 | MAE Loss | Score   |                                                    |
| --------------------- | -------- | ------- | -------------------------------------------------- |
| DecisionTreeRegressor | 17.03343 | 0.05545 | Criterion: mse                                     |
|                       | 17.01315 | 0.05551 | Criterion: friedman_mse                            |
|                       | 16.92268 | 0.05580 | Criterion: mae                                     |
|                       | 20.86998 | 0.04572 | Criterion: Poisson                                 |
| ExtraTreeRegressor    | 16.99063 | 0.05558 | An extremely randomized tree regressor.<br />'mse' |
|                       | 16.99415 | 0.05557 | Criterion: friedman_mse                            |
|                       | 16.98082 | 0.05561 | Criterion: mae                                     |

## Ensemble Methods

| Model                     | MAE Loss     | Score       |                                           |
| ------------------------- | ------------ | ----------- | ----------------------------------------- |
| AdaBoostRegressor         | 18.71484     | 0.05072     | Base_estimator: DT<br />n_estimators: 8   |
| BaggingRegressor          | 16.73922     | 0.05637     | Base_estimator: DT<br />n_estimators: 10  |
|                           | *15.95616\** | *0.05898\** | *n_estimators: 50<br />10-Fold training*  |
| ExtraTreesRegressor       | 16.19455     | 0.05816     | n_estimators: 100                         |
|                           | *16.10310*   | *0.05847*   | *n_estimators: 100<br />10-Fold training* |
| GradientBoostingRegressor | 15.81050     | 0.05949     | subsample = 1.0,<br />n_est = 100         |
|                           | 15.80639     | 0.05950     | subsample = 0.5; lr=0.1                   |
|                           | 15.60075     | 0.06024     | Subsample = 1; lr=0.2<br />10_Fold        |
|                           | *15.53368\** | *0.06048\** | *Subsample = 0.5; lr=0.2<br />10_Fold*    |
| RandomForestRegressor     | 15.94809     | 0.05900     |                                           |

![AdaBoost Score vs Num-Estimators](/Users/kakusou/Library/Application Support/typora-user-images/image-20210509115156211.png)AdaBoost Score vs Num-Estimators

![Bagging vs #Estimators](/Users/kakusou/Library/Application Support/typora-user-images/image-20210509120342902.png)Bagging vs #Estimators

![ExtraTrees vs #Estimators](/Users/kakusou/Library/Application Support/typora-user-images/image-20210509123213129.png)ExtraTrees vs #Estimators

## Lightgbm

| Model         | MAE Loss | Score   |               |
| ------------- | -------- | ------- | ------------- |
| LGBMRegressor | 15.23600 | 0.06159 | Default       |
|               | 15.23326 | 0.06160 | reg_alpha=0.6 |

## XGBoost

| Model            | MAE Loss | Score |      |
| ---------------- | -------- | ----- | ---- |
| XGBoostRegressor |          |       |      |

## CatBoost

| Model             | MAE Loss | Score   |                                                              |
| ----------------- | -------- | ------- | ------------------------------------------------------------ |
| CatBoostRegressor | 15.43482 | 0.06085 | 'depth': 7, 'learning_rate': 0.8, 'l2_leaf_reg': 2, 'num_boost_round': 10000, 'random_seed': 94, |
|                   | 15.36688 | 0.06110 | 'depth': 6, 'learning_rate': 0.8, 'l2_leaf_reg': 2,          |
|                   |          |         |                                                              |

