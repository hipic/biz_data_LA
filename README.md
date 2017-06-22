# HiPIC Data Science

**Section 1. Business Data Analysis near CalStateLA, USC, UCLA using Spark SQL in Los Angeles, 2016**
: Business Data Set is collected from Google Local and Yelp API by the graduate students, Sridhar reddy Puli (spuli@calstatela.edu), Ram Dharan Donda (rdonda@calstatela.edu), Goutham kumar Pola (gpola@calstatela.edu) and Vinay Chennupati (vchennu@calstatela.edu) at Dept of Computer Information Systems, California State University Los Angeles on Summer 2016
Business Categories reviewed frequently near CalStateLA, USC, UCLA by [HiPIC](http://web.calstatela.edu/centers/hipic/) of CalStateLA under Prof [Jongwook Woo](http://web.calstatela.edu/faculty/jwoo5/)'s guidance.

**Section 2. Predicting popularity of Yelp Business using Spark Machine Learning, 2017**
: This Section is a part of Machine Learning Project by [Ruchi Singh](https://www.linkedin.com/in/ruchi-singh-68015945/) and [Ovanes H. Mikaelian](https://www.linkedin.com/in/hovik-mikaelian-93a257a3/) under the guidance of Prof [Jongwook Woo](http://web.calstatela.edu/faculty/jwoo5/) at Dept of Computer Information Systems, California State University Los Angeles on Spring 2017.

##### **Section 2.1. Applying Collaborative Filtering to Create Recommendations for a Business Category to Each User, 2017**
: This Section is a part of Machine Learning Project by graduate student [Ovanes H. Mikaelian](https://www.linkedin.com/in/hovik-mikaelian-93a257a3/) under the guidance of Prof [Jongwook Woo](http://web.calstatela.edu/faculty/jwoo5/) at Dept of Computer Information Systems, California State University Los Angeles in Spring 2017.

##### **Section 2.2 Clustering Food related Business in Yelp using Spark Machine Learning, 2017**
: This Section is a part of Machine Learning Project by [Ruchi Singh](https://www.linkedin.com/in/ruchi-singh-68015945/) under the guidance of Prof [Jongwook Woo](http://web.calstatela.edu/faculty/jwoo5/) at Dept of Computer Information Systems, California State University Los Angeles on Spring 2017.

# Section 1. Business Data Analysis near CalStateLA, USC, UCLA using Spark SQL in Los Angeles, 2016
Business data in Los Angeles is collected using Yelp and Google Local API and it is analyzed per near CalStateLA, USC, UCLA using Hadoop and Spark

Business Categories reviewed frequently near CalStateLA, USC, UCLA.
![Image of Result Map](https://github.com/hipic/biz_data_LA/blob/master/nearCampusBizSchools.JPG)

Tutorial pdf file: biz_analysis_LA_v4.2.pdf

ipynb Code in Spark: biz_analysis_LA.ipynb

**This tutorial is added to [Databricks](http://www.databricks.com)'s [Databricks Training](https://docs.databricks.com/spark/latest/training/cal-state-la-biz-data-la.html).**

# Section 2. Predicting popularity of Yelp Business using Spark Machine Learning, 2017

To predict the popularity of the business we defined the popular business to have stars greater than 3 and unpopular business to have stars less than 3. To select the feature columns and have the accurate prediction for the popularity of the business we chose the food category. All the attribute columns related to the food category like good for breakfast, lunch, dinner, take out, delivery, parking, alcohol, Wi-Fi, waiter service, wheelchair and noise level are considered as feature columns. We categorize all the columns for the classification models like Two Class Logistic regression and Two Class Boosted Decision Tree.

The logistic regression is used to find the probability of the two states of the target variable. Whereas the boosted decision tree is an ensemble learning tree to make the prediction. Both the models are suitable for the prediction of the popularity of the business. The logistic regression is considered to be the best in fast training and linear classification model whereas boosted decision tree is known for its accuracy, fast training and large memory footprint, apt for big data.

The detail implementation of two class logistic regression in Spark ML using Train Validation Split and Binary Classification Evaluator in Spark can be followed in [Classification-Food.md](https://github.com/hipic/biz_data_LA/blob/master/Spark%20ML-%20Classification/Classification-Food.md) . The regularization para meters used to avoid the imbalances in data are 0.01, 0.5 and 2.0. The PramGridBuilder is used to generate all possible combinations of regularization parameter, max iterator and threshold. The AUR value of model is low due to training the logistic regression for the complete dataset as oppose to the sampled dataset. The dataset had a very small percentage of food business having less than 3 stars. Thus, the model is not trained well to predict unpopular business. The result could improve if the dataset was balanced with popular and unpopular business.

Spark Code: [Classification-Food.md](https://github.com/hipic/biz_data_LA/blob/master/Spark%20ML-%20Classification/Classification-Food.md)

## Section 2.1. Applying Collaborative Filtering to Create Recommendations for a Business Category to Each User, 2017

The goal of the recommender is to provide Yelp users with recommendations for business categories based on their previous business ratings, as well as the business ratings of other users. The model has a feature to predict the future ratings by user for a category.

Initially, in Azure, SQL transformation was conducted to select the average number of stars that each user has given to a category. The new dataset was saved under the name "reviewstar.csv" and transferred to Spark.

The cleaned and transformed data contains four columns: user_id, category, review_count, stars. User_id and category are selected as features and stars is selected as label. The dataset is split to train and test fractions by .7 to .3 ratio. We have used ALS (Alternating Least Squares) algorithm to build the recommender. Also, we’ve defined parameters and used fit method to train the model. Then we test the model to predict the stars that a particular user will give to a particular category. Based on that prediction, we recommend a category to each user. 

Spark Code: [Collaborative+Filtering+Recommender+Tutorial.md](https://github.com/hipic/biz_data_LA/blob/master/Spark%20ML%20-%20Recommender/Collaborative%2BFiltering%2BRecommender%2BTutorial%20.md)

## Section 2.2. Clustering Food related Business in Yelp using Spark Machine Learning, 2017

Cluster analysis divides data into groups (clusters) that are meaningful, useful or both. Clustering can be a useful starting point for other purposes like data summarization and can be used for statistics, patter recognition, information retrieval, machine learning and data mining.

Yelp is widely used to review and check reviews, mostly for services and food business. We decided to work with the food related business and filtered the dataset for such business. The food data includes feature columns like stars, review_count, and categorized columns describing whether the restaurant is good for breakfast, dinner, lunch or take-out. The latter attributes describing the food business are chosen as features. The model clusters the food business into 5 cluster with appropriate center points, based on the count of reviews each business has received. For example, review_count from 1 to 80 were put in one cluster, 81-200 in another, etc. By comparing the distance from center in SparkML, we got a good result in SparkML. The maximum distance of the the data points from the center is 9.77.

Spark Code: [Cluster-Food.md](https://github.com/hipic/biz_data_LA/blob/master/Spark%20ML%20-%20Clustering/Cluster-Food.md)
