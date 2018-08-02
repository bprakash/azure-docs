# Databricks notebook source
# MAGIC %md #Power Plant ML Pipeline Application
# MAGIC This is an end-to-end example of using a number of different machine learning algorithms to solve a supervised regression problem.
# MAGIC 
# MAGIC ###Table of Contents
# MAGIC 
# MAGIC - *Step 1: Business Understanding*
# MAGIC - *Step 2: Load Your Data*
# MAGIC - *Step 3: Explore Your Data*
# MAGIC - *Step 4: Visualize Your Data*
# MAGIC - *Step 5: Data Preparation*
# MAGIC - *Step 6: Data Modeling*
# MAGIC - *Step 7: Tuning and Evaluation*
# MAGIC - *Step 8: Deployment*
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *We are trying to predict power output given a set of readings from various sensors in a gas-fired power generation plant.  Power generation is a complex process, and understanding and predicting power output is an important element in managing a plant and its connection to the power grid.*
# MAGIC 
# MAGIC More information about Peaker or Peaking Power Plants can be found on Wikipedia https://en.wikipedia.org/wiki/Peaking_power_plant
# MAGIC 
# MAGIC 
# MAGIC Given this business problem, we need to translate it to a Machine Learning task.  The ML task is regression since the label (or target) we are trying to predict is numeric.
# MAGIC 
# MAGIC 
# MAGIC The example data is provided by UCI at [UCI Machine Learning Repository Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
# MAGIC 
# MAGIC You can read the background on the UCI page, but in summary we have collected a number of readings from sensors at a Gas Fired Power Plant
# MAGIC (also called a Peaker Plant) and now we want to use those sensor readings to predict how much power the plant will generate.
# MAGIC 
# MAGIC 
# MAGIC More information about Machine Learning with Spark can be found in the [Spark MLLib Programming Guide](https://spark.apache.org/docs/latest/mllib-guide.html)
# MAGIC 
# MAGIC 
# MAGIC * Please note this example only works with Spark version 1.4 or higher*
# MAGIC * Ref: https://demo.cloud.databricks.com/#notebook/282283

# COMMAND ----------

# MAGIC %md ##Step 1: Business Understanding
# MAGIC The first step in any machine learning task is to understand the business need. 
# MAGIC 
# MAGIC As described in the overview we are trying to predict power output given a set of readings from various sensors in a gas-fired power generation plant.
# MAGIC 
# MAGIC The problem is a regression problem since the label (or target) we are trying to predict is numeric

# COMMAND ----------

# MAGIC %md ##Step 2: Load Your Data
# MAGIC Now that we understand what we are trying to do, we need to load our data and describe it, explore it and verify it.

# COMMAND ----------

# MAGIC %md
# MAGIC Since the dataset is relatively small, we will use the upload feature in Databricks to upload the data as a table.
# MAGIC 
# MAGIC First download the Data Folder from [UCI Machine Learning Repository Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
# MAGIC 
# MAGIC The file is a multi-tab Excel document so you will need to save each tab as a Text file export. 
# MAGIC 
# MAGIC I prefer exporting as a Tab-Separated-Values (TSV) since it is more consistent than CSV.
# MAGIC 
# MAGIC Call each file Folds5x2_pp<Sheet 1..5>.tsv and save to your machine.
# MAGIC 
# MAGIC Go to the Databricks Menu > Tables > Create Table
# MAGIC 
# MAGIC Select Datasource as "File"
# MAGIC 
# MAGIC Upload *ALL* 5 files at once.
# MAGIC 
# MAGIC See screenshots below:
# MAGIC 
# MAGIC 
# MAGIC **2.1. Create Table**
# MAGIC   _________________
# MAGIC 
# MAGIC When you import your data, name your table `power_plant`, specify all of the columns with the datatype `Double` and make sure you check the `First row is header` box.
# MAGIC 
# MAGIC ![alt text](http://training.databricks.com/databricks_guide/1_4_ML_Power_Plant_Import_Table.png)
# MAGIC 
# MAGIC **2.2. Review Schema**
# MAGIC   __________________
# MAGIC 
# MAGIC Your table schema and preview should look like this after you click ```Create Table```:
# MAGIC 
# MAGIC ![alt text](http://training.databricks.com/databricks_guide/1_4_ML_Power_Plant_Import_Table_Schema.png)

# COMMAND ----------

# MAGIC %md #### Step 2: (Load your Data Alternative Option):
# MAGIC 
# MAGIC If you did Step 2 already you can skip down to Step 3. If you want to skip the data import and just load the data from our public datasets in Azure Storage use the cell below.

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/

# COMMAND ----------

from pyspark.sql.types import *

customSchema = StructType([ \
    StructField("AT", DoubleType(), True), \
    StructField("V", DoubleType(), True), \
    StructField("AP", DoubleType(), True), \
    StructField("RH", DoubleType(), True), \
    StructField("PE", DoubleType(), True)])

df = sqlContext.read.format('csv').\
        options(header = 'true', delimiter='\t').\
        load("/databricks-datasets/power-plant/data/", schema = customSchema)

display(df)

# COMMAND ----------

sqlContext.sql("DROP TABLE IF EXISTS power_plant")
dbutils.fs.rm("dbfs:/user/hive/warehouse/power_plant", True)

df.write.saveAsTable("power_plant")

# COMMAND ----------

# MAGIC %md ##Step 3: Explore Your Data
# MAGIC Now that we understand what we are trying to do, we need to load our data and describe it, explore it and verify it.

# COMMAND ----------

# MAGIC %sql 
# MAGIC --We can use %sql to query the rows
# MAGIC 
# MAGIC SELECT * FROM power_plant

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESC power_plant

# COMMAND ----------

# MAGIC %md **Schema Definition**
# MAGIC 
# MAGIC Our schema definition from UCI appears below:
# MAGIC 
# MAGIC - AT = Atmospheric Temperature in C
# MAGIC - V = Exhaust Vaccum Speed
# MAGIC - AP = Atmospheric Pressure
# MAGIC - RH = Relative Humidity
# MAGIC - PE = Power Output
# MAGIC 
# MAGIC PE is our label or target. This is the value we are trying to predict given the measurements.
# MAGIC 
# MAGIC *Reference [UCI Machine Learning Repository Combined Cycle Power Plant Data Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)*

# COMMAND ----------

# MAGIC %md Let's do some basic statistical analysis of all the columns. 
# MAGIC 
# MAGIC We can use the describe function with no parameters to get some basic stats for each column like count, mean, max, min and standard deviation.  More information can be found in the [Spark API docs](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrame)

# COMMAND ----------

display(sqlContext.table("power_plant").describe())

# COMMAND ----------

# MAGIC %md ##Step 4: Visualize Your Data
# MAGIC 
# MAGIC To understand our data, we will look for correlations between features and the label.  This can be important when choosing a model.  E.g., if features and a label are linearly correlated, a linear model like Linear Regression can do well; if the relationship is very non-linear, more complex models such as Decision Trees can be better. We use the Databricks built in visualization to view each of our predictors in relation to the label column as a scatter plot to see the correlation between the predictors and the label.

# COMMAND ----------

# MAGIC %sql select AT as Temperature, PE as Power from power_plant

# COMMAND ----------

# MAGIC %md It looks like there is strong linear correlation between temperature and Power Output

# COMMAND ----------

# MAGIC %sql select V as ExhaustVaccum, PE as Power from power_plant;

# COMMAND ----------

# MAGIC %md The linear correlation is not as strong between Exhaust Vacuum Speed and Power Output but there is some semblance of a pattern.

# COMMAND ----------

# MAGIC %sql select AP Pressure, PE as Power from power_plant;

# COMMAND ----------

# MAGIC %sql select RH Humidity, PE Power from power_plant;

# COMMAND ----------

# MAGIC %md ...and atmospheric pressure and relative humidity seem to have little to no linear correlation

# COMMAND ----------

# MAGIC %md ##Step 5: Data Preparation
# MAGIC 
# MAGIC The next step is to prepare the data. Since all of this data is numeric and consistent, this is a simple task for us today.
# MAGIC 
# MAGIC We will need to convert the predictor features from columns to Feature Vectors using the org.apache.spark.ml.feature.VectorAssembler
# MAGIC 
# MAGIC The VectorAssembler will be the first step in building our ML pipeline.

# COMMAND ----------

from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

dataset = sqlContext.table("power_plant")
vectorizer = VectorAssembler(
    inputCols=["AT", "V", "AP", "RH"],
    outputCol="features")

output = vectorizer.transform(dataset)

print(output.select("features", "AT").first())

# COMMAND ----------

# MAGIC %md ##Step 6: Data Modeling
# MAGIC Now let's model our data to predict what the power output will be given a set of sensor readings
# MAGIC 
# MAGIC Our first model will be based on simple linear regression since we saw some linear patterns in our data based on the scatter plots during the exploration stage.

# COMMAND ----------

# First let's hold out 20% of our data for testing and leave 80% for training
(split20, split80) = dataset.randomSplit([0.20, 0.80], 1800009193L)

# COMMAND ----------

# Let's cache these datasets for performance
testSet = split20.cache()
trainingSet = split80.cache()
print "Test count: ", testSet.count()
print "Training count: ", trainingSet.count()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression()

print lr.explainParams()

# COMMAND ----------

# MAGIC %md The cell below is based on the Spark ML pipeline API. More information can be found in the Spark ML Programming Guide at https://spark.apache.org/docs/latest/ml-guide.html

# COMMAND ----------

from pyspark.ml import Pipeline

# Now we set the parameters for the method
lr.setPredictionCol("Predicted_PE")\
  .setLabelCol("PE")\
  .setMaxIter(100)\
  .setRegParam(0.1)

# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()
lrPipeline.setStages([vectorizer, lr])
# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSet)

# COMMAND ----------

# MAGIC %md 
# MAGIC Since Linear Regression is simply a line of best fit over the data that minimizes the square of the error, given multiple input dimensions we can express each predictor as a line function of the form:
# MAGIC 
# MAGIC \\(y = a + b x_1 + b x_2 + b x_i ...  \\)
# MAGIC 
# MAGIC where a is the intercept and b are coefficients.
# MAGIC 
# MAGIC To express the coefficients of that line we can retrieve the Estimator stage from the PipelineModel and express the weights and the intercept for the function.

# COMMAND ----------

# The intercept is as follows:
modelStage = lrModel.stages[1]
print modelStage.intercept

# COMMAND ----------

modelStage.coefficients.values

# COMMAND ----------

# The coefficents (i.e. weights) are as follows:
import math

weights = modelStage.coefficients.values

featuresNoLabel = [col for col in dataset.columns if col != "PE"]

coefficients = sc.parallelize(weights).zip(sc.parallelize(featuresNoLabel))

# Now let's sort the coefficients from the largest to the smallest

equation = "y = %s " % modelStage.intercept

for (weight, name) in coefficients.sortByKey().collect():
  if (weight > 0):
    equation += " + (%s X %s)" % (weight, name)
  else:
    equation += " - (%s X %s)" % (abs(weight), name)
    
print "Linear Regression Equation: " + equation

# COMMAND ----------

# MAGIC %md Based on examining the output it shows there is a strong negative correlation between Atmospheric Temperature (AT) and Power Output.
# MAGIC 
# MAGIC But our other dimenensions seem to have little to no correlation with Power Output. Do you remember **Step 2: Explore Your Data**? When we visualized each predictor against Power Output using a Scatter Plot, only the temperature variable seemed to have a linear correlation with Power Output so our final equation seems logical.
# MAGIC 
# MAGIC 
# MAGIC Now let's see what our predictions look like given this model.

# COMMAND ----------

predictionsAndLabels = lrModel.transform(testSet)

display(predictionsAndLabels.select("AT", "V", "AP", "RH", "PE", "Predicted_PE"))

# COMMAND ----------

# MAGIC %md Now that we have real predictions we can use an evaluation metric such as Root Mean Squared Error to validate our regression model. The lower the Root Mean Squared Error, the better our model.

# COMMAND ----------

from pyspark.mllib.evaluation import RegressionMetrics

valuesAndPreds = predictionsAndLabels.select("Predicted_PE", "PE").rdd.map(lambda x: (x.__getitem__('Predicted_PE'), x.__getitem__('PE')))

# Instantiate metrics object
metrics = RegressionMetrics(valuesAndPreds)

# Squared Error
print("RMSE = %s" % metrics.rootMeanSquaredError)
# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)
# R-squared
print("R-squared = %s" % metrics.r2)

# COMMAND ----------

# MAGIC %md Generally a good model will have 68% of predictions within 1 RMSE and 95% within 2 RMSE of the actual value. Let's calculate and see if a RMSE of 4.51 meets this criteria.

# COMMAND ----------

# First we calculate the residual error and divide it by the RMSE
predictionsAndLabels.selectExpr("PE", "Predicted_PE", "PE - Predicted_PE Residual_Error", """ (PE - Predicted_PE) / %s Within_RSME""" % metrics.rootMeanSquaredError ).registerTempTable("Power_Plant_RMSE_Evaluation")

# COMMAND ----------

# MAGIC %sql SELECT * from Power_Plant_RMSE_Evaluation

# COMMAND ----------

# MAGIC %sql -- Now we can display the RMSE as a Histogram. Clearly this shows that the RMSE is centered around 0 with the vast majority of the error within 2 RMSEs.
# MAGIC SELECT Within_RSME  from Power_Plant_RMSE_Evaluation

# COMMAND ----------

# MAGIC %sql SELECT case when Within_RSME <= 1.0 and Within_RSME >= -1.0 then 1 when  Within_RSME <= 2.0 and Within_RSME >= -2.0 then 2 else 3 end RSME_Multiple, COUNT(*) count  from Power_Plant_RMSE_Evaluation
# MAGIC group by case when Within_RSME <= 1.0 and Within_RSME >= -1.0 then 1 when  Within_RSME <= 2.0 and Within_RSME >= -2.0 then 2 else 3 end

# COMMAND ----------

# MAGIC %md So we have 68% of our training data within 1 RMSE and 97% (68% + 29%) within 2 RMSE. So the model is pretty decent. Let's see if we can tune the model to improve it further.

# COMMAND ----------

# MAGIC %md #Step 7: Tuning and Evaluation
# MAGIC 
# MAGIC Now that we have a model with all of the data let's try to make a better model by tuning over several parameters.
# MAGIC 
# MAGIC Documentation Available here: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.tuning

# COMMAND ----------

regParam = [i / 100.0 for i in range(1,11)]
regParam

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

regEval = RegressionEvaluator(predictionCol="Predicted_PE")
regEval.setLabelCol("PE")\
  .setMetricName("rmse")
  
regParam = [i / 100.0 for i in range(1,11)]

grid = ParamGridBuilder().addGrid(lr.regParam, regParam).build()
  
crossval = CrossValidator(estimator=lrPipeline, estimatorParamMaps=grid, evaluator = regEval, numFolds=5)

cvModel = crossval.fit(trainingSet)

# COMMAND ----------

# MAGIC %md Now that we have tuned let's see what we got for tuning parameters and what our RMSE was versus our intial model

# COMMAND ----------

predictionsAndLabels = cvModel.transform(testSet)

valuesAndPreds = predictionsAndLabels.select("Predicted_PE", "PE").rdd.map(lambda x: (x.__getitem__('Predicted_PE'), x.__getitem__('PE')))
metrics = RegressionMetrics(valuesAndPreds)

rmse = metrics.rootMeanSquaredError
explainedVariance = metrics.explainedVariance
r2 = metrics.r2

# Squared Error
print("RMSE = %s" % rmse)
# Explained variance
print("Explained variance = %s" % explainedVariance)
# R-squared
print("R-squared = %s" % r2)

# COMMAND ----------

# MAGIC %md So our initial untuned and tuned linear regression models are statistically identical.
# MAGIC 
# MAGIC Given that the only linearly correlated variable is Temperature, it makes sense try another machine learning method such a Decision Tree to handle non-linear data and see if we can improve our model
# MAGIC 
# MAGIC A Decision Tree creates a model based on splitting variables using a tree structure. We will first start with a single decision tree model.
# MAGIC 
# MAGIC Reference Decision Trees: https://en.wikipedia.org/wiki/Decision_tree_learning

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Decision Tree Models

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.setLabelCol("PE")
dt.setPredictionCol("Predicted_PE")
dt.setFeaturesCol("features")
dt.setMaxBins(100)

dtPipeline = Pipeline()
dtPipeline.setStages([vectorizer, dt])
# Let's just resuse our CrossValidator

crossval.setEstimator(dtPipeline)

paramGrid = ParamGridBuilder()\
  .addGrid(dt.maxDepth, range(2, 8))\
  .build()
crossval.setEstimatorParamMaps(paramGrid)

dtModel = crossval.fit(trainingSet)

# COMMAND ----------

# MAGIC %md Now let's see how our DecisionTree model compares to our LinearRegression model

# COMMAND ----------

predictionsAndLabels = dtModel.bestModel.transform(testSet)

valuesAndPreds = predictionsAndLabels.select("Predicted_PE", "PE").rdd.map(lambda x: (x.__getitem__('Predicted_PE'), x.__getitem__('PE')))
metrics = RegressionMetrics(valuesAndPreds)

rmse = metrics.rootMeanSquaredError
explainedVariance = metrics.explainedVariance
r2 = metrics.r2

# Squared Error
print("RMSE = %s" % rmse)
# Explained variance
print("Explained variance = %s" % explainedVariance)
# R-squared
print("R-squared = %s" % r2)

# COMMAND ----------

# MAGIC %md So our DecisionTree was slightly better than our LinearRegression model (LR: 4.58 vs DT: 3.86).

# COMMAND ----------

# MAGIC %md Run predictions on new unlabeled data coming in from the power plant

# COMMAND ----------

newData = [(1.81,39.42,1026.92,76.97),(3.2,41.31,997.67,98.84),(3.38,41.31,998.79,97.76)]
newDataDf = spark.createDataFrame(newData, ['AT','V','AP','RH'])
display(newDataDf)

# COMMAND ----------

predictionsNewDataDf = dtModel.bestModel.transform(newDataDf)
display(predictionsNewDataDf)

# COMMAND ----------

# MAGIC %md Export the model out to be used in a Java application, ref [ML Model Export](https://docs.azuredatabricks.net/spark/latest/mllib/index.html#databricks-ml-model-export)

# COMMAND ----------

from dbmlModelExport import ModelExport

# COMMAND ----------

dbutils.fs.rm( "/tmp/power_plant_ml_pipeline", recurse=True)

# COMMAND ----------

ModelExport.exportModel(dtModel.bestModel, "/tmp/power_plant_ml_pipeline")

# COMMAND ----------

# MAGIC %fs ls /tmp/power_plant_ml_pipeline

# COMMAND ----------

# MAGIC %fs rm /FileStore/power_plant_ml_pipeline.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC cp -r /dbfs/tmp/power_plant_ml_pipeline* /tmp/
# MAGIC zip -r /tmp/power_plant_ml_pipeline.zip /tmp/power_plant_ml_pipeline*
# MAGIC cp /tmp/power_plant_ml_pipeline.zip /dbfs/Filestore/.

# COMMAND ----------

# MAGIC %sh
# MAGIC zip -r /dbfs/FileStore/power_plant_ml_pipeline.zip /dbfs/tmp/power_plant_ml_pipeline*

# COMMAND ----------

# MAGIC %fs ls /FileStore/power_plant_ml_pipeline.zip

# COMMAND ----------

# MAGIC %md  Get a link to the downloadable zip via:
# MAGIC `https://[MY_DATABRICKS_URL]/files/[FILE_NAME].zip`.  E.g., if you access Databricks at `https://eastus2.azuredatabricks.net?o=1111111111111111`, then your link would be:
# MAGIC `https://eastus2.azuredatabricks.net/files/power_plant_ml_pipeline.zip?o=1111111111111111`.