
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/></a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

#### Author: [Ruchi Singh](https://www.linkedin.com/in/ruchi-singh-68015945/)

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/20/2017

## Clustering
Cluster analysis divides data into groups (clusters) that are meaningful, useful or both. It describes the objects and their relationships. K-means clustering is a partitional clustering technique that attemps to find user specified numbers of cluster (K), which are representd by their centroids.

## Clustering of food related business in Yelp
Grouping food related business based on their review count, taking in account appropriate feature columns.

## Download Data

Download the "Business-Food.csv" file and upload in Databricks. Data-> default-> Create Table. Rename the table as "Food2" and check for all the columns datatype. 

This is the data to be used for training the machine learning algorithm.


```python
### Import the Libraries
You will use the **KMeans** class to create your model. This will require a vector of features, so you will also use the **VectorAssembler** class.

```


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
```

### Load Source Data
The source data Business-Food is a comma-separated values (CSV) file, and incldues the following features:
- review_count: The number of reviews for the business
- Take-out: If the food business has take facility  
- GoodFor-lunch: The customer's think the food place is good for lunch
- GoodFor-dinner: The customer's think the food place is good for dinner
- GoodFor-breakfast: The customer's think the food place is good for breakfast
- stars: The star rating given by the customers for the food business (1-5)


```python
# Adopt shcmea to read csv data set in the schema. 

csv = sqlContext.sql("Select * from food2")
```


```python
data = csv.select("review_count","Take-out", "GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","stars")
```


```python
data.show(5)
```

### String Indexer

StringIndexer encodes a string column of labels to a column of label indices.


```python
def indexStringColumns(df, cols):
    #variable newdf will be updated several times
    newdata = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-x")
        sm = si.fit(newdata)
        newdata = sm.transform(newdata).drop(c)
        newdata = newdata.withColumnRenamed(c+"-x", c)
    return newdata

dfnumeric = indexStringColumns(data, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast"])
```

### Hot Encoder

One-hot encoding maps a column of label indices to a column of binary vectors, with at most a single one-value.


```python
def oneHotEncodeColumns(df, cols):
    from pyspark.ml.feature import OneHotEncoder
    newdf = df
    for c in cols:
        onehotenc = OneHotEncoder(inputCol=c, outputCol=c+"-onehot", dropLast=False)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    return newdf

dfhot = oneHotEncodeColumns(dfnumeric, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast"])
```

### Create the K-Means Model
You will use the feaures in the food business data to create a K-Means model with a k value of 5. This will be used to generate 5 clusters.


```python
assembler = VectorAssembler(inputCols = list(set(dfhot.columns) | set(['stars','review_count'])), outputCol="features")
train = assembler.transform(dfhot)
knum = 5
kmeans = KMeans(featuresCol=assembler.getOutputCol(), predictionCol="cluster", k=knum, seed=0)
model = kmeans.fit(train)
print "Model Created!"
```

### Get the Cluster Centers
The cluster centers are indicated as vector coordinates.


```python
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
```

### Predict Clusters
Now that we have trained the model, we can use it to segemnt the customer data into 5 clusters and show each business with their allocated cluster.


```python
# data set does not need to be divided to train and test
prediction = model.transform(train)
prediction.groupBy("cluster").count().orderBy("cluster").show()
```


```python
# Look at the features of each cluster

# define dictionary
customerCluster = {}
for i in range(0,knum):
    tmp = prediction.select("stars","review_count","Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast")\
                                    .where("cluster =" +  str(i))
    customerCluster[str(i)]= tmp
    print "Cluster"+str(i)
    customerCluster[str(i)].show(3)
```
