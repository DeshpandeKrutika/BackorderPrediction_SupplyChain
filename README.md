# Optimizing Backorder Prediction on Imbalanced Data using ML Pipelines and GridSearch

### Data

This dataset was originally posted on Kaggle. The key task is to predict whether a product/part will go on backorder.

Product backorder may be the result of strong sales performance (e.g. the product is in such a high demand that production cannot keep up with sales). However, backorders can upset consumers, lead to canceled orders and decreased customer loyalty. Companies want to avoid backorders, but also avoid overstocking every product (leading to higher inventory costs).

This dataset has ~1.9 million observations of products/parts in an 8 week period. The source of the data is unreferenced.

Outcome: whether the product went on backorder
Predictors: Current inventory, sales history, forecasted sales, recommended stocking amount, product risk flags etc. (22 predictors in total)
The features and the target variable of the dataset are as follows:

#### Description:

*  Features: 
sku - Random ID for the product
national_inv - Current inventory level for the part
lead_time - Transit time for product (if available)
in_transit_qty - Amount of product in transit from source
forecast_3_month - Forecast sales for the next 3 months
forecast_6_month - Forecast sales for the next 6 months
forecast_9_month - Forecast sales for the next 9 months
sales_1_month - Sales quantity for the prior 1 month time period
sales_3_month - Sales quantity for the prior 3 month time period
sales_6_month - Sales quantity for the prior 6 month time period
sales_9_month - Sales quantity for the prior 9 month time period
min_bank - Minimum recommend amount to stock
potential_issue - Source issue for part identified
pieces_past_due - Parts overdue from source
perf_6_month_avg - Source performance for prior 6 month period
perf_12_month_avg - Source performance for prior 12 month period
local_bo_qty - Amount of stock orders overdue
deck_risk - Part risk flag
oe_constraint - Part risk flag
ppap_risk - Part risk flag
stop_auto_buy - Part risk flag
rev_stop - Part risk flag

*  Target 
went_on_backorder - Product actually went on backorder

### Project Steps:

* data preparation and exploratory data analysis
* anomaly detection / removal
* dimensionality reduction and then
* train and validate

Developing Pipeline:
  Anomaly detection
  Dimensionality Reduction
  Train a classification model

I used 3 pipelines to compare the performance:

 #### Pipeline I
  - Anomaly detection (Elliptic Envelope)
  - Dimensionality reduction (SelectKBest)
  - Model training/validation (Gradient Boosting Classifier)

#### Pipeline II
  - Anomaly detection (Isolation Forest)
  - Dimensionality reduction (SelectFromModel- LinearSVC)
  - Model training/validation (Random Forest)

#### Pipeline III
  - Anomaly detection (Local Outlier Factor)
  - Dimensionality reduction (PCA)
  - Model training/validation (SVC)

### Conclusion:

Preprocessing the unseen data:

  Basic EDA was done on the test data, similar to Part 1. 
  Columns that were evidently irrelevant were removed. 'sku' was the unique identifier of the dataset which is not required.'perf_6_mont_avg' and 'perf_12_month_avg' seemed irrelevant and ambigous. The performance values ranged from -0.99 to 1.0 which was ambigous to interpret.

  Missing values were handled by using the median value for the feature 'lead_time' with over 14,000 missing values. There was   one row which had misiing values in almost all the columns, that row was eliminated from the dataset.

Model Performance:

  The model used was the best model from Part 2. [Isolation Forest + SelectFromModel(LinearSVC) + Random Forest Classifier]
  
  Hyperparamters: 
    1) IsolationForest(contamination=0.08)
    2) Pipeline(steps=[('Lsvc',
                 SelectFromModel(estimator=LinearSVC(dual=False,
                                                     penalty='l1'))),
                ('rf',
                 RandomForestClassifier(max_depth=20, max_features='sqrt',
                                        n_estimators=600))])
                                 
  The accuracy of the model on training data (sampled from Part 1) is 98%, with recall of 99% and F1 score of 98%.
  The model has high precision, recall and F1 score for both the classes on the sampled data.
  
  On the unseen data, the overall model accuracy obtained is 89%, with recall of 81% and F1 score of 14.1%.
  
  The model does extremely well in predicting the majority class with precision of 100% and recall value of 89%.
  The model has a high recall value of 81% for the minority class but poor precision of 8%.
  
Minority Class: (Went to backorder- yes)
  
  For every 100 instances for the minority class, 81 were predicted correctly. So we see that the model does well in predicting the minority class (when the product goes for backorder). However, it is predicting much more samples as the minority class (went to backorder- yes), than the actual number of samples that did go to backorder. That is, there is higher number of false positives. For every 100 predicted as positive (went to backorder- yes), only 8 were true, is actually went to backorder. (went to backorder- yes).
  
  The model is sensitive to backordered items, having a high true positive rate. [Sensitivity is the porbability of a positive   result conditioned on the individual/sample truly being positive]. The items taht should go to backorder were correctly predicted.
  However, it has lower specificity (True Negative Rate) which is the probability of a negative result conditioned on the individual/sample truly being negative]. This means that there items that didnt actually go to backorder were predicted to go into backorder.
  
  There is usually a trade-ff between sensitivity and specificity, such that higher sensitivities will mean lower specificities (and vice versa) which is what we can observe with above model.

### Brief Summary for Management

[Assumptions: (1) The business sells items that are critical for the function or sustainence of fast-paced industry (eg, medical equipments  (2) The business has competitiors and wants to stay ahead of the curve to meet the market's high demand.]


                                                  Problem:

Understanding the need of the business to resolve the issue of product availability in situations of high demand, it is crucial to higlight the problems that come with backorderering items. If the company consistently sees items in backorder, 
 - it could be taken as a signal that the company's operations are too lean
 - it could also mean that the company is losing out on business by not providing the products demanded by its customers
 - if a customer sees products on backorder—and notices this frequently—they may decide to cancel orders, forcing the company to issue refunds and readjust their books.
 - if the expected wait time until the product becomes available is long, customers may look elsewhere for a substitute, that is, loss of customer to competition
 - it may require additional resources in managing pre-orders or clients that are waiting for their product.
 - it could lead to eventual loss of market share as customers become frustrated with the lack of product availability
 - due to improper inventory management, increase in overhead costs such as logistics and public communication




                                            Model and proposed solution:

Understanding above problems and consequences of the lack of product availability, a machine learning model is proposed to help predict if an item is needed to be stocked up in the inventory, ie, go for backorder or not. This is an efficient way to guage the demand and the current state of supply chain operations for the company.

The model provides an accuracy of 89% which means that it acurately predicts 89% of the times, whether an item must go to backorder or not. 

However, it is grasped that the issue is to better the availability of the product that is expected to meet a high demand or is currently insufficient in amount at the company inventory. Hence, the model proposed is one that is sensitive to such items. 
The recall score for the cases where the items are subjected to backorder is 81% which means that out of 100 items, 81 are rightly identified which are required to be restocked/backordered. This is a good estimator for backorder items and there is a low chance of missing those items which need to be restocked. This will save the company from running out of supply for such items. Meeting the customer demand at the promised time would build better customer trust and inturn customer loyalty to the company. This will reduce the overhead costs and unnecessary resource planning for order cancellations, refunds, etc. With higher customer satisfaction, the market share could increase with increased order numbers. This could directly affect revenue generation positively, but also give the company the scope to increase their prices with respect to competition as they can promise delivery in time.

This model does,however, have a trade-off in precision, of predicting the items which are wrongly classified as backordered when they did not require so. The precision rate of the model is 8% which means that out of 100 items predicted to be backordered, only 8 actually went to backorder.This could translate to increased supply to the inventory for products that may not have as significant demand as projected. This could come with a cost. The products would require more marketing and resources to manage selling before expiry of itmes and in some cases also produce more storage and transportation cost. This can be avoided by judging the current market trends or knowledge of the product history. This may also be corrected by estimating the following orders considering inflation of inventory. 

The ultimate decision of using the model defintely depends on the consequence and gravity of missing products that should require to be backordered, or made available but also depend on the tolerance of having to inflate the inventory. However, this model can be trusted to not miss the items with the genuine need and demand. 

The best way to make use of this model would be to flag the items predicted as positive (for backorder), cautiously order the items based on order history and other knowledge while constantly monitoring the flagged products of changing needs of customers and market trends.

