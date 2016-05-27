# Databricks notebook source exported at Fri, 27 May 2016 08:43:12 UTC
"""
Author: Santiago Morante
"""

from pyspark.sql import DataFrame as DFPyspark
from pyspark.rdd import RDD
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pandas import DataFrame as DFPandas
from numpy.linalg import norm


class DMF():
    """
    DISTRIBUTED ANOMALY DETECTION ALGORITHM BASED ON DISSIMILARITY MAPPING FILTERING
    S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and feature selection for robot learning," 
    Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on, Seoul, 2015, pp. 428-433.
    doi: 10.1109/HUMANOIDS.2015.7363569
    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7363569&isnumber=7362951

    :param data:                 RDD of data points
    :param dissimilarityMethod:  Method for performing dissimilarity
    :param mappingMethod:        Method for performing mapping
    :param filteringMethod:      Method for performing filtering
    :param theta:                Threhold used in filtering step
    """
   
    def __init__(self, dissimilarityMethod="euclidean", mappingMethod="sum", filteringMethod="zscore", theta=0):
      self.dissimilarityMethod = dissimilarityMethod
      self.mappingMethod = mappingMethod
      self.filteringMethod = filteringMethod
      self.theta=theta

    def convertToRDD(self, dataset):
      sc = SparkContext.getOrCreate()
      if isinstance(dataset, RDD):
        return dataset
      
      elif isinstance(dataset, DFPyspark):
          return dataset.rdd.flatMap(list)
      
      elif isinstance(dataset, DFPandas):
        sqlContext = SQLContext.getOrCreate(sc)
        return sqlContext.createDataFrame(dataset).rdd.flatMap(list)
      
      else:
        try:
            return sc.parallelize(dataset)
        except TypeError:
            print("convertToRDD cannot convert your dataset because it is not one of the allowed types (RDD, dataframe (sql) or dataframe(pandas))")
 
      
    def dissimilarity(self, doublePairs): 
      """DISSIMILARITY: calculate distance between elements"""
      if self.dissimilarityMethod == "euclidean":
        return doublePairs[0][0], norm(doublePairs[0][1]-doublePairs[1][1])
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
        return (pairs[1] - meanData)/float(stdevData) > theta 
      else:
         raise ValueError("filteringMethod should be an allowed method, "
                              "but got %s." % str(filteringMethod))

    def detect(self, dataset):
      """Detects anomalies in the dataset"""
 
      #convert to RDD
      rdd = self.convertToRDD(dataset)
      
      #index data
      dataIndex = rdd.zipWithIndex().map(lambda x: (x[1],x[0]))
      
      #dissimilarity
      dataDissimilarity = dataIndex.cartesian(dataIndex).map(self.dissimilarity)
      
      #mapping
      dataMapping = dataDissimilarity.reduceByKey(self.mapping)
      
      #filtering
      meanData= dataMapping.values().mean()
      stdevData= dataMapping.values().stdev()
      dataFiltering = dataMapping.filter(lambda pairs: self.filtering(pairs, meanData, stdevData, self.theta))
      
      #return
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

#DATA: generate sine wave with outliers
data = sin(range(0,10)).tolist() + \
                    [35] +  \
                    sin(range(11,20)).tolist() + \
                    [32]  +  \
                    sin(range(21,30)).tolist() + \
                    [60]  +  \
                    sin(range(31,50)).tolist() 

#plot
x = linspace(0, len(data))
fig, ax = plt.subplots()
plt.plot(x, data, '-', linewidth=2)
display(fig)

# COMMAND ----------

#parallelize data
dataParallel = sc.parallelize(data)    

#create model
model= DMF()

#detect using model
print ("Index, totalValue: ", model.detect(data).collect())

# COMMAND ----------


