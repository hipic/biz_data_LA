
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>


#### Author: [Ovanes H. Mikaelian](https://www.linkedin.com/in/hovik-mikaelian-93a257a3/)
<p><img align="center" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

------

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/18/2017

## Collaborative Filtering
Collaborative filtering is a machine learning technique that predicts ratings awarded to items by users.

### Import the ALS class
In this module, we use the Alternating Least Squares collaborative filtering algorithm to creater a recommender.


```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.sql import functions as F

```

### Load Source Data
The source data for the recommender the reviewstar.csv which we created in Microsoft Azure. It contains numeric data on users, categories, review_count, and stars. Based in the file we created a table in Databricks.


```python

stars1 = spark.sql("select * from reviewstar")
stars1.show(20)

```

### Prepare the Data
To prepare the data, split it into a training set and a test set.


```python
data = stars1.select("user_id", "categories", "stars")
splits = data.randomSplit([0.7, 0.3])
train = splits[0].withColumnRenamed("stars", "label")
test = splits[1].withColumnRenamed("stars", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print "Training Rows:", train_rows, " Testing Rows:", test_rows
```

### Build the Recommender
In ALS, users and categories are described by a small set of latent features (factors) that can be used to predict missing entries.

We can use the features to produce some sort of algorithm (**ALS**) to intelligently calculate stars given by each user to a particular business category.

The ALS class is an estimator, so we used its **fit** method to traing a model. It could also be included in a pipeline. Rather than specifying a feature vector and as label, the ALS algorithm requries a numeric user ID, categories, and stars.


```python
als = ALS(userCol="user_id", itemCol="categories", ratingCol="label")
```


```python
#### Add paramGrid and Validation
```


```python
paramGrid = ParamGridBuilder() \
                    .addGrid(als.rank, [1, 5]) \
                    .addGrid(als.maxIter, [5, 10]) \
                    .addGrid(als.regParam, [0.3, 0.1, 0.01]) \
                    .addGrid(als.alpha, [2.0,3.0]) \
                    .build()


```


```python
cv = TrainValidationSplit(estimator=als, evaluator=RegressionEvaluator(), 
                          estimatorParamMaps=paramGrid, trainRatio=0.8)
model = cv.fit(train)
```

### Test the Recommender
Now that we've trained the recommender, we can see how accurately it predicts known star ratings in the test set.


```python
prediction = model.transform(test)

# Remove NaN values from prediction (due to SPARK-14489) [1]
prediction = prediction.filter(prediction.prediction != float('nan'))

# Round floats to whole numbers
prediction = prediction.withColumn("prediction", F.abs(F.round(prediction["prediction"],0)))

prediction.select("user_id", "categories", "prediction", "trueLabel").show(100, truncate=False)
```

#### RegressionEvaluator
Calculate RMSE using RegressionEvaluator


```python
# RegressionEvaluator: predictionCol="prediction", metricName="rmse"
evaluator = RegressionEvaluator(labelCol="trueLabel", 
                                predictionCol="prediction", 
                                metricName="rmse")
rmse = evaluator.evaluate(prediction)
print "Root Mean Square Error (RMSE):", rmse
```

### ALS model in implicit type
If the rating matrix is derived from another source of information (i.e. it is inferred from other signals), you can set implicitPrefs to True to get better results. 

Build and Train ALS model with "implicitPrefs=True"


```python
als_implicit = ALS(userCol="user_id", itemCol="categories", ratingCol="label", implicitPrefs=True)
#als_implicit = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="label", implicitPrefs=True)
#model_implicit = als_implicit.fit(train)
```


```python
paramGrid = ParamGridBuilder() \
                    .addGrid(als_implicit.rank, [1, 5]) \
                    .addGrid(als_implicit.maxIter, [5, 10]) \
                    .addGrid(als_implicit.regParam, [0.3, 0.1, 0.01]) \
                    .addGrid(als_implicit.alpha, [2.0,3.0]) \
                    .build()

```


```python
cv = TrainValidationSplit(estimator=als_implicit, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
#cv = CrossValidator(estimator=als_implicit, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator())
model_implicit = cv.fit(train)
```


```python
prediction_implicit = model_implicit.transform(test)

# Remove NaN values from prediction (due to SPARK-14489) [1]
prediction_implicit = prediction_implicit.filter(prediction_implicit.prediction != float('nan'))

# Round floats to whole numbers
prediction_implicit = prediction_implicit.withColumn("prediction", F.abs(F.round(prediction_implicit["prediction"],0)))


prediction_implicit.select("user_id", "categories", "prediction", "trueLabel").show(100, truncate=False)
```


```python
# RegressionEvaluator: predictionCol="prediction", metricName="rmse"
evaluator_implicit = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse_implicit = evaluator_implicit.evaluate(prediction_implicit)
print "ImplicitRoot Mean Square Error (RMSE):", rmse_implicit
```

The data used in this exercise describes 5-star rating activity from Yelp. It was created by Yelp for the Yelp Data Challenge competition.

This Yelp datasets are publicly available for download at <https://www.yelp.com/dataset_challenge/dataset>.

**Reference**
1. Predicting Song Listens Using Apache Spark, https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3175648861028866/48824497172554/657465297935335/latest.html


```python

```
