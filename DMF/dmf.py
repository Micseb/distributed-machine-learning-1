"""
Author: Santiago Morante
DISSIMILARITY MAPPING FILTERING
"""

###################################################################################
###########################   CLASS DMF   #########################################
###################################################################################
from fastdtw import fastdtw
from numpy.linalg import norm
from numpy import array, count_nonzero

class DMF():
    """
    DISSIMILARITY MAPPING FILTERING:
    S. Morante, J. G. Victores and C. Balaguer, "Automatic demonstration and feature selection for robot learning," 
    Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on, Seoul, 2015, pp. 428-433.
    doi: 10.1109/HUMANOIDS.2015.7363569
    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7363569&isnumber=7362951
    
    :param dissimilarityMethod:  Method for performing dissimilarity: euclidean (default), dtw, hamming
    :param mappingMethod:        Method for performing mapping: sum (default)
    :param filteringMethod:      Method for performing filtering: zscore (default), threshold
    :param alpha:                Threshold used in filtering step (zero by default)

    Use case (assuming "signals" is in correct format):
      >>> model= DMF(dissimilarityMethod="euclidean", filteringMethod="zscore")
      >>> print ("Index, totalValue: ", model.detect(signals).collect())    
    """
   
    def __init__(self, dissimilarityMethod="euclidean", mappingMethod="sum", filteringMethod="zscore", alpha=0):
      """Initialize parameters"""
      self.dissimilarityMethod = dissimilarityMethod
      self.mappingMethod = mappingMethod
      self.filteringMethod = filteringMethod
      self.alpha = alpha

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

