"""
Author: Santiago Morante
SIGNAL CLASS (INDEXED DATA VECTOR)
"""

###################################################################################
###########################   CLASS SIGNAL ########################################
###################################################################################
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame as DFPyspark
from pyspark.rdd import RDD
from pandas import DataFrame as DFPandas

class Signal():
  """
  SIGNAL: 
  The input data must be in regular table style (rows, columns): 
    1. First, it is converted into RDD (if not yet)
    2. Every element is cast to string (if categorical) or float (if numerical) and missing values are imputed
    3. Every row is indexed with the index in the first position of a tuple (index, listOfValuesOfTheRow)
  
  Each tuple in a Signal instance represents an indexed vector of data.
  
  :param typeOfData:           Defines type of data: "categorical" or "numerical"
  :param missingNumDefault:    Number to insert in missing values for numerical datasets
  :param missingCatDefault:    string to insert in missing values for categorical datasets
  
  Use case:
    >>> data=sqlContext.read.table("your_table")
    >>> signals = Signal(typeOfData="numerical").create(data)
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
