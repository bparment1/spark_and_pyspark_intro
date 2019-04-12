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
#DATE MODIFIED: 04/11/2019
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
in_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/Exercise_4/data"
#in_dir = "/nfs/public-data/training"
#ARGS 2
out_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/Exercise_4/outputs"
#out_dir = "/home/participant32"
#out_dir = "/resarch-home/bparmentier"

#ARGS 3:
create_out_dir=True #create a new ouput dir if TRUE
#ARGS 4
out_suffix = "exercise4_04112019" #output suffix for the files and ouptut folder
#ARGS 5
data_fname = 'r_variables_harris_county_exercise4_02072019.txt'
#ARGS 19
prop = 0.3 #proportion of observations for hold-out/testing
#ARGS 20
random_seed = 100 #random seed for reproducibility

################# START SCRIPT ###############################

## seting up SPARK
sc= SparkContext()
sqlContext = SQLContext(sc)
#https://www.guru99.com/pyspark-tutorial.html
#dataset = pd.read_csv("data/AS/test_v2.csv")
#sc = SparkContext(conf=conf)
#sqlCtx = SQLContext(sc)
#sdf = sqlCtx.createDataFrame(dataset)

######### PART 0: Set up the output dir ################


#set up the working directory
#Create output directory

if create_out_dir==True:
    out_dir_new = "output_data_"+out_suffix
    out_dir = os.path.join(out_dir,out_dir_new)
    create_dir_and_check_existence(out_dir)
    os.chdir(out_dir)        #set working directory
else:
    os.chdir(create_out_dir) #use working dir defined earlier
    


### Let's read in the information that contains variables
data_df = pd.read_csv(os.path.join(in_dir,data_fname))
data_df.columns
data_df.head()

################
##### Step 1: Prepare categorical features/covariates by rescaling values

## Relevant variables used:
selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
selected_target_names = ['change'] #also called dependent variable

## We need to account for categorical versus continuous variables
selected_categorical_var_names=['land_cover']
selected_continuous_var_names=list(set(selected_covariates_names) - set(selected_categorical_var_names))
##Find frequency of unique values:
freq_val_df = data_df[selected_categorical_var_names].apply(pd.value_counts)
print(freq_val_df.head())
values_cat = array(data_df[selected_categorical_var_names].values) #note this is assuming only one cat val here

### Let's read in the information that contains variables
data_df = pd.read_csv(os.path.join(in_dir,data_fname))
data_df.columns
data_df.head()

################
##### Step 1: Prepare categorical features/covariates by rescaling values

## Relevant variables used:
selected_covariates_names = ['land_cover', 'slope', 'roads_dist', 'developped_dist']
selected_target_names = ['change'] #also called dependent variable

## We need to account for categorical versus continuous variables
selected_categorical_var_names=['land_cover']
selected_continuous_var_names=list(set(selected_covariates_names) - set(selected_categorical_var_names))
##Find frequency of unique values:
freq_val_df = data_df[selected_categorical_var_names].apply(pd.value_counts)
print(freq_val_df.head())
values_cat = array(data_df[selected_categorical_var_names].values) #note this is assuming only one cat val here

label_encoder = LabelEncoder()  # labeling categories
one_hot_encoder = OneHotEncoder(sparse=False) #generate dummy variables
### First integer encode:
integer_encoded = label_encoder.fit_transform(values_cat)
print(integer_encoded)
# Binary encode:
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)

#33 generate dummy variables
onehot_encoded = one_hot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
onehot_encoded.shape
type(onehot_encoded)

#Check values generated: invert to check value?
onehot_encoded[0:5,]
values_cat[0:5,]
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[1,:])])
print(inverted)

#assign back to the data.frame
unique_val = np.array(freq_val_df.index)
unique_val = np.sort(unique_val)
print(unique_val)
names_cat = ['lc_' + str(i) for i in unique_val]
print(names_cat)
onehot_encoded_df = pd.DataFrame(onehot_encoded,columns=names_cat)
onehot_encoded_df.columns
onehot_encoded_df.head()
onehot_encoded_df.shape
data_df.shape

## Add the new encoded variables to the data frame
data_df= pd.concat([data_df,onehot_encoded_df],sort=False,axis=1)
data_df.shape
data_df.head()

selected_covariates_names_updated = selected_continuous_var_names + names_cat 

##############
## Step 2: Split training and testing and rescaling for continuous variables

##############
## Step 2: Split training and testing and rescaling for continuous variables

X_train, X_test, y_train, y_test = train_test_split(data_df[selected_covariates_names_updated], 
                                                    data_df[selected_target_names], 
                                                    test_size=prop, 
                                                    random_state=random_seed)
X_train.shape
X_train.head()

#### Scaling between 0-1 for continuous variables
# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))
### need to select only the continuous var:
scaled_training = scaler.fit_transform(X_train[selected_continuous_var_names])
scaled_testing = scaler.transform(X_test[selected_continuous_var_names])
type(scaled_training) # array
scaled_training.shape

## Concatenate column-wise
X_testing_df = pd.DataFrame(np.concatenate((X_test[names_cat].values,scaled_testing),axis=1),
                                            columns=names_cat+selected_continuous_var_names)

X_training_df = pd.DataFrame(np.concatenate((X_train[names_cat].values,scaled_training),axis=1),
                                            columns=names_cat+selected_continuous_var_names)

X_training_df.head()

###########################################
### PART IV: Run model and perform assessment ###########################

####################
###### Step 1: fit glm logistic model and generate predictions

X_y_training_df = pd.DataFrame(np.concatenate((X_training_df.values,y_train),axis=1),
                                            columns=list(X_training_df)+['change'])

training_spark_df = sqlContext.createDataFrame(X_y_training_df)
training_spark_df
#https://www.guru99.com/pyspark-tutorial.html
#https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a

vectorAssembler = VectorAssembler(inputCols = list(X_training_df), 
                                  outputCol = 'features')
vtraining_df = vectorAssembler.transform(training_spark_df)
vtraining_df = vtraining_df.select(['features', 'change'])
vtraining_df.show(3)

#vectorAssembler = VectorAssembler(inputCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT'],
#                                  outputCol = 'features')
#vhouse_df = vectorAssembler.transform(house_df)
#vhouse_df = vhouse_df.select(['features', 'MV'])
#vhouse_df.show(3)


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'feaes', labelCol='MV', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
tur

# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(featuresCol='features',labelCol='change',maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(vtraining_df)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrMoodel.intercept))


###################### END OF SCRIPT #####################










                   






