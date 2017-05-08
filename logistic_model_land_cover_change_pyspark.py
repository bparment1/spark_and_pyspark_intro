# -*- coding: utf-8 -*-
"""
Spyder Editor.
"""
#################################### Land Use and Land Cover Change #######################################
############################ Analyze Land Cover change in Houston #######################################
#This script performs analyses to test pyspark g aggregated NLCD values.
#The goal is to assess land cover change using two land cover maps in the Houston areas.
#Additional datasets are provided for the land cover change modeling. A model is built for Harris county.
#
#AUTHORS: Benoit Parmentier
#DATE CREATED: 01/07/2019
#DATE MODIFIED: 05/07/2019
#Version: 1
#PROJECT: AAG 2019 Geospatial Short Course
#TO DO:
#
#COMMIT: changes to modeling
#
#################################################################################################
	
	
###### Library used in this script

import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import rasterio
import subprocess
import pandas as pd
import os, glob
from rasterio import plot
import geopandas as gpd
import descartes
import pysal as ps
from cartopy import crs as ccrs
from pyproj import Proj
from osgeo import osr
from shapely.geometry import Point
from collections import OrderedDict
import webcolors
import copy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from pyspark.ml.classification import LogisticRegression
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler

#from pyspark import SparkConf, SparkContext
#from pyspark.sql import SQLContext

from pyspark.sql import SparkSession

################ NOW FUNCTIONS  ###################

##------------------
# Functions used in the script 
##------------------

def create_dir_and_check_existence(path):
    #Create a new directory
    try:
        os.makedirs(path)
    except:
        print ("directory already exists")

############################################################################
#####  Parameters and argument set up ########### 

#ARGS 1
in_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/pyspark/data"
#ARGS 2
out_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/pyspark/outputs"
#out_dir = "/home/participant32"
#out_dir = "/resarch-home/bparmentier"
#in_dir = "/nfs/public-data/training"
#
#ARGS 3:
create_out_dir=True #create a new ouput dir if TRUE
#ARGS 4
out_suffix = "pyspark_application_land_change_05072019" #output suffix for the files and ouptut folder
#ARGS 5
data_fname = 'r_variables_harris_county_land_change_modeling_05072019.txt'
#ARGS 19
prop = 0.3 #proportion of observations for hold-out/testing
#ARGS 20
random_seed = 100 #random seed for reproducibility

################# START SCRIPT ###############################


## seting up SPARK
#sc= SparkContext()
#import findspark as fs
#fs.init()
#spark = SparkSession.builder.appName('Land_change_example').getOrCreate()
sc = SparkSession.builder.appName('Land_change_example').getOrCreate()

#sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

######### PART 0: Read in data ################
    
data_df = sc.read.csv(os.path.join(in_dir,data_fname), 
                      header=True, inferSchema=True)

data_df.show(5)
#schema on read!
data_df.printSchema()
data_df.count()
data_df.columns
data_df.describe().show()

data_df.groupby('change').count().show()
data_df.groupby('change').mean('slope').show()


################Feature engineering
##### Step 1: Prepare categorical features/covariates by rescaling values


## Relevant variables used:
selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
selected_target_names = ['change'] #also called dependent variable

## We need to account for categorical versus continuous variables
selected_categorical_var_names=['land_cover']
selected_continuous_var_names=list(set(selected_covariates_names) - set(selected_categorical_var_names))

##Find frequency of unique values: do this with SQL
#freq_val_df = data_df[selected_categorical_var_names].apply(pd.value_counts)
#print(freq_val_df.head())
#values_cat = array(data_df[selected_categorical_var_names].values) #note this is assuming only one cat val here

#https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
#categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

selected_categorical_var_names=['land_cover']
stages = [] # stages in our Pipeline
    
for catCol in selected_categorical_var_names:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=catCol, 
                                  outputCol=catCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], 
                                     outputCols=[catCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
    

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="change", outputCol="label")
stages += [label_stringIdx]

#selected_covariates_names_updated = selected_continuous_var_names + names_cat 

##############
## Step 2: Split training and testing and rescaling for continuous variables

training_df, test_df = data_df.randomSplit([1-prop,prop],seed=random_seed)

#https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c


### NEEED SCALING NOW

#### Scaling between 0-1 for continuous variables
# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.

vectorAssembler = VectorAssembler(inputCols = selected_covariates_names_updated, 
                                  outputCol = 'features')
vtraining_df = vectorAssembler.transform(training_spark_df)

from pyspark.ml.feature import StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
standardscaler=standardscaler.fit(training_df)
training_df = standardscaler.transform(training_df)
testing_df=standardscaler.transform(tesing_df)

#raw_data.select("features","Scaled_features").show(5)

###########################################
### PART IV: Run model and perform assessment ###########################

training_spark_df = sqlContext.createDataFrame(X_y_training_df)

#https://www.guru99.com/pyspark-tutorial.html
#https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a

vectorAssembler = VectorAssembler(inputCols = selected_covariates_names_updated, 
                                  outputCol = 'features')
vtraining_df = vectorAssembler.transform(training_spark_df)

#vectorAssembler = VectorAssembler(inputCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT'],
#                                  outputCol = 'features')
#vhouse_df = vectorAssembler.transform(house_df)
#vhouse_df = vhouse_df.select(['features', 'MV'])
#vhouse_df.show(3)
vtraining_df.show(3)

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='MV', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

#https://www.guru99.com/pyspark-tutorial.html
#dataset = pd.read_csv("data/AS/test_v2.csv")
#sc = SparkContext(conf=conf)
#sqlCtx = SQLContext(sc)
#sdf = sqlCtx.createDataFrame(dataset)

# Load training data
#training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

### Need to add the labels
#https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html

#featuresCol = 'features', labelCol='MV'

# Fit the model
lrModel = lr.fit(training)
lrModel = lr.fit(vtraining_df)
#
#
#lfModel = lr.fit(X_train.values,y_train.values.ravel())

model_logistic = LogisticRegression() #instantiate model object
model_logistic = model_logistic.fit(X_train.values,y_train.values.ravel())

print("model coefficients: ",model_logistic.coef_)
selected_covariates_names_updated

pred_test_prob = model_logistic.predict_proba(X_test.values)
y_scores_test = pred_test_prob[:,1]
pred_train_prob = model_logistic.predict_proba(X_train.values)
y_scores_train = pred_train_prob[:,1]

### Note that we only have about 10% change in the dataset so setting 50% does not make sense!!
sum(data_df.change)/data_df.shape[0]
sum(y_train.change)/y_train.shape[0]
sns.set(color_codes=True) #improves layout with bar and background grid
sns.countplot(x='change',data=data_df)
plt.show()
plt.savefig('count_plot')

# Explore values distribution
f, ax = plt.subplots(1, 2)
sns.distplot(y_scores_train,ax=ax[0])#title='January residuals')
sns.distplot(y_scores_test,ax=ax[1])#title='January residuals')
ax[0].set(title="Predicted training probabilities") 
ax[1].set(title="Predicted testing probabilities") 

####################
###### Step 2: Model assessment with ROC and AUC

#Compute AUC
auc_val_train =roc_auc_score(y_train,y_scores_train)
auc_val_test =roc_auc_score(y_test,y_scores_test)

print("AUC train: ", auc_val_train)
print("AUC test: ", auc_val_test)

#Generate inputs for ROC curves
fpr, tpr, thresholds = roc_curve(y_test, 
                                 y_scores_test)
plt.figure()
plt.plot(fpr, tpr, 
         label='Logistic Regression (area = %0.2f)' % auc_val_test)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

###################### END OF SCRIPT #####################










                   






