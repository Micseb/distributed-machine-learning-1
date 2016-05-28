# Databricks notebook source exported at Sat, 28 May 2016 11:16:47 UTC
"""
Author: Santiago Morante
DISTRIBUTED ANOMALY DETECTION ALGORITHM BASED ON DISSIMILARITY MAPPING FILTERING
S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and feature selection for robot learning," 
Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on, Seoul, 2015, pp. 428-433.
doi: 10.1109/HUMANOIDS.2015.7363569
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7363569&isnumber=7362951
"""

from pyspark.sql import DataFrame as DFPyspark
from pyspark.sql import SQLContext
from pyspark.rdd import RDD
from pyspark import SparkContext
from pandas import DataFrame as DFPandas
from numpy.linalg import norm
from fastdtw import fastdtw

class Signal():
  """
  Signal:
  1. Takes data in regular table style (rows, columns), 
  2. Converts it to RDD (if not yet)
  3. Indexes every row with the index in the first position of a tupla (index, listOfValuesOfTheRow),
  4. Imputes values in missing positions. --toDo
  
  Each tupla in a Signal instance represents a vector of data.
  """

  def create(self, dataset):
    """Creates Signal from list, rdd, dataframe (spark) or dataframe (pandas)"""
    rdd = self.convertToRDD(dataset)
    return self.indexRDD(rdd)
    
  def convertToRDD(self, dataset):
    """Converts dataset into RDD (if not yet)"""
    sc = SparkContext.getOrCreate()
    if isinstance(dataset, RDD):
      return dataset
    elif isinstance(dataset, DFPyspark):
        return dataset.rdd.map(list)
    elif isinstance(dataset, DFPandas):
      sqlContext = SQLContext.getOrCreate(sc)
      return sqlContext.createDataFrame(dataset).rdd.map(list)
    else:
      try:
          return sc.parallelize(dataset)
      except TypeError:
          print("convertToRDD cannot convert dataset because it is not one of the allowed types \
          (RDD, dataframe (sparkSQL) or dataframe (pandas))")
  
  def indexRDD(self, rdd):
    """Index each row of a RDD as a tuple (index, listOfValuesOfTheRow)"""
    return rdd.zipWithIndex().map(lambda x: (x[1],x[0]))
  
#######################################################################  

class DMF():
    """
    DISSIMILARITY MAPPING FILTERING:
    S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and feature selection for robot learning," 
    Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on, Seoul, 2015, pp. 428-433.
    
    :param data:                 RDD of data points
    :param dissimilarityMethod:  Method for performing dissimilarity
    :param mappingMethod:        Method for performing mapping
    :param filteringMethod:      Method for performing filtering
    :param theta:                Threhold used in filtering step
    """
   
    def __init__(self, dissimilarityMethod="euclidean", mappingMethod="sum", filteringMethod="zscore", theta=0):
      """Initialize parameters"""
      self.dissimilarityMethod = dissimilarityMethod
      self.mappingMethod = mappingMethod
      self.filteringMethod = filteringMethod
      self.theta = theta

    def dissimilarity(self, doublePairs): 
      """DISSIMILARITY: calculate distance between elements"""
      #euclidean
      if self.dissimilarityMethod == "euclidean":
        #convert to float
        element0 = [float(x) for x in doublePairs[0][1]]
        element1 = [float(x) for x in doublePairs[1][1]]
        #subtract lists
        subtraction = [a - b for a, b in zip(element1, element0)]
        return float(doublePairs[0][0]), norm(subtraction)
      #dtw
      if self.dissimilarityMethod == "dtw":
        #convert to float
        element0 = [float(x) for x in doublePairs[0][1]]
        element1 = [float(x) for x in doublePairs[1][1]]
        #fastdtw
        distance, path = fastdtw(element0, element1)
        return float(doublePairs[0][0]), distance
      #unknown
      else:
        raise ValueError("dissimilarityMethod should be an allowed method, "
                            "but got %s." % str(dissimilarityMethod))

    def mapping(self, a,b):
      """MAPPING: reduce the comparisons matrix to single value per element key"""
      if self.mappingMethod == "sum":
        return a+b
      else:
        raise ValueError("mappingMethod should be an allowed method, "
                              "but got %s." % str(mappingMethod))
        
    def filtering(self, pairs, meanData, stdevData, theta):
      """FILTERING: filtering data by threshold (theta) using Z-score"""
      if self.filteringMethod == "zscore":
        try:
          return (pairs[1] - meanData)/float(stdevData) > theta 
        except:
          raise ValueError("Filtering Z-score division by zero (StDev=0)!  (all values are equal)")
      else:
         raise ValueError("filteringMethod should be an allowed method, "
                              "but got %s." % str(filteringMethod))

    def detect(self, signals):
      """Detects anomalies in the dataset (must be in format (index,listOfValues))"""
      #dissimilarity
      dataDissimilarity = signals.cartesian(signals).map(self.dissimilarity)
      #mapping
      dataMapping = dataDissimilarity.reduceByKey(self.mapping)
      #filtering
      meanData= dataMapping.values().mean()
      stdevData= dataMapping.values().stdev()
      dataFiltering = dataMapping.filter(lambda pairs: self.filtering(pairs, meanData, stdevData, self.theta))
      return dataFiltering

# COMMAND ----------

#######################################################
### MAIN ##############################################
#######################################################

from numpy import sin, linspace
import matplotlib.pylab as plt
from pyspark import SparkContext

#sparkContext
sc = SparkContext.getOrCreate()
sqlContext = SQLContext.getOrCreate(sc)
data=sqlContext.read.table("dim2small")


# COMMAND ----------

#create model
model= DMF(dissimilarityMethod="dtw")
signals = Signal().create(data)

#detect using model
print ("Index, totalValue: ", model.detect(signals).collect())

# COMMAND ----------


