# Databricks notebook source exported at Sat, 28 May 2016 15:07:55 UTC
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
  SIGNAL: 
  The input data must be in regular table style (rows, columns): 
    1. First, it is converted into RDD (if not yet)
    2. Every element is cast to string (if categorical) or float (if numerical) and missing values are imputed
    3. Every row is indexed with the index in the first position of a tuple (index, listOfValuesOfTheRow)
  
  Each tuple in a Signal instance represents an indexed vector of data.
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
      except:
          raise TypeError("convertToRDD cannot convert dataset because it is not in a recognized format!")
          
  
  def imputeMissingValues(self, line):
    """Imputes default value to missing values in each line of RDD"""
    if self.typeOfData == "numerical":
      return [float(x) if x else self.missingNumDefault for x in line]
    elif self.typeOfData == "categorical":
      return [str(x) if x else self.missingCatDefault for x in line]
    else:
      raise NameError("typeOfData not recognized")
      
  
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
        raise NameError("dissimilarityMethod not recognized")

    def mapping(self, a,b):
      """MAPPING: reduce the comparisons matrix to single value per element key"""
      if self.mappingMethod == "sum":
        return a+b
      else:
        raise NameError("mappingMethod not recognized")
        
    def filtering(self, pairs):
      """FILTERING: filtering data by threshold (alpha) using Z-score"""
      if self.filteringMethod == "zscore":
        try:
          return (pairs[1] - self.mappingMean)/float(self.mappingStDev) > self.alpha 
        except:
          print("[WARNING] Filtering Z-score division by zero (StDev=0)!  (all values are equal)")
          return []
      elif self.filteringMethod == "threshold":
        return pairs[1] > self.alpha
      else:
        raise NameError("filteringMethod not recognized")

    def detect(self, signals):
      """Detects anomalies in the dataset (must be in format (index,listOfValues))"""
      #dissimilarity
      dataDissimilarity = signals.cartesian(signals).map(self.dissimilarity)
      #mapping
      dataMapping = dataDissimilarity.reduceByKey(self.mapping)
      #filtering
      if self.filteringMethod == "zscore":
        self.mappingMean  = dataMapping.values().mean()
        self.mappingStDev = dataMapping.values().stdev()
      dataFiltering = dataMapping.filter(self.filtering)
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
data=sqlContext.read.table("dim2smallmissing")

#create signal
signals = Signal(typeOfData="numerical").create(data)

#create model
model= DMF(dissimilarityMethod="euclidean", filteringMethod="zscore")

#detect using model
print ("Index, totalValue: ", model.detect(signals).collect())

# COMMAND ----------


