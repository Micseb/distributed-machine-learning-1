# Databricks notebook source exported at Sat, 28 May 2016 14:25:05 UTC
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
from numpy import array, count_nonzero
from fastdtw import fastdtw


class Signal():
  """
  Signal:
  1. Takes data in regular table style (rows, columns), 
  2. Converts it to RDD (if not yet)
  3. Indexes every row with the index in the first position of a tuple (index, listOfValuesOfTheRow),
  4. Converts every element to float and imputes missing values
  
  Each tuple in a Signal instance represents a vector of data.
  """

  def __init__(self, typeOfData="numerical", missingNumDefault=0, missingCatDefault="NA"):
        """Initialize parameters"""
        self.typeOfData = typeOfData
        self.missingNumDefault = missingNumDefault
        self.missingCatDefault = missingCatDefault
      
  def create(self, dataset):
    """Creates Signal from list, rdd, dataframe (spark) or dataframe (pandas)"""
    rdd = self.convertToRDD(dataset).map(self.imputeMissingValues)
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
  
  def imputeMissingValues(self, line):
    """Imputes default value to missing values in each line of RDD"""
    if self.typeOfData == "numerical":
      return [float(x) if x else self.missingNumDefault for x in line]
    elif self.typeOfData == "categorical":
      return [str(x) if x else self.missingCatDefault for x in line]
    else:
      raise ValueError("typeOfData not recognized")
  
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
    :param alpha:                Threshold used in filtering step
    """
   
    def __init__(self, dissimilarityMethod="euclidean", mappingMethod="sum", filteringMethod="zscore", alpha=0):
      """Initialize parameters"""
      self.dissimilarityMethod = dissimilarityMethod
      self.mappingMethod = mappingMethod
      self.filteringMethod = filteringMethod
      self.alpha = alpha

    def dissimilarity(self, doublePairs): 
      """DISSIMILARITY: calculate distance between elements"""
      if self.dissimilarityMethod == "euclidean":
        return doublePairs[0][0], norm(array(doublePairs[1][1])-array(doublePairs[0][1]))
      elif self.dissimilarityMethod == "dtw":
        distance, path = fastdtw(doublePairs[0][1], doublePairs[1][1])
        return doublePairs[0][0], distance
      elif self.dissimilarityMethod == "hamming":
        return doublePairs[0][0], count_nonzero(doublePairs[0][1] != doublePairs[1][1])
      else:
        raise ValueError("dissimilarityMethod not recognized")

    def mapping(self, a,b):
      """MAPPING: reduce the comparisons matrix to single value per element key"""
      if self.mappingMethod == "sum":
        return a+b
      else:
        raise ValueError("mappingMethod not recognized")
        
    def filtering(self, pairs, meanData, stdevData):
      """FILTERING: filtering data by threshold (alpha) using Z-score"""
      if self.filteringMethod == "zscore":
        try:
          return (pairs[1] - meanData)/float(stdevData) > self.alpha 
        except:
          print("[WARNING] Filtering Z-score division by zero (StDev=0)!  (all values are equal)")
          return []
      elif self.filteringMethod == "threshold":
        return pairs[1] > self.alpha
      else:
        raise ValueError("filteringMethod not recognized")

    def detect(self, signals):
      """Detects anomalies in the dataset (must be in format (index,listOfValues))"""
      #dissimilarity
      dataDissimilarity = signals.cartesian(signals).map(self.dissimilarity)
      #mapping
      dataMapping = dataDissimilarity.reduceByKey(self.mapping)
      #filtering
      meanData= dataMapping.values().mean()
      stdevData= dataMapping.values().stdev()
      dataFiltering = dataMapping.filter(lambda pairs: self.filtering(pairs, meanData, stdevData))
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
data=sqlContext.read.table("dim3medium")

# COMMAND ----------

#create model
model= DMF(dissimilarityMethod="hamming", filteringMethod="threshold", alpha=13)
signals = Signal(typeOfData="categorical").create(data)
#detect using model
print ("Index, totalValue: ", model.detect(signals).collect())

# COMMAND ----------


