"""
Author: Santiago Morante
DISTRIBUTED MULTIOBJECTIVE OPTIMIZATION BASED ON EVOLUTIVE ALGORITHMS
"""

###################################################################################
########################### CLASS EVOL POP ########################################
###################################################################################
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame as DFPyspark
from pyspark.rdd import RDD
from pandas import DataFrame as DFPandas
from numpy.linalg import norm
from random import SystemRandom

class EvolPop():
  """
  EVOLPOP: 
  The input data must be a list, a Dataframe or a RDD. When "getBestIndividual" is called the following steps are executed: 
    1. First, the input data (called prototype) is converted into RDD (if not yet)
    2. Every element is used as a reference to generate a mutated population of individuals
    3. For each individual it is computed the costs associated to its paramters, a the distances to the optimal values (each feature independently) 
    4. A pairwise selection is performed using pareto dominance, and the best individual is returned
  
  :param prototype:             Used as reference to generate a population (usually a single element)
  :param variables_type         Lists contating the type of each variable (only accepts "numerical" o "categorical")
  :param ranges:                Vector of ranges to define bounds in value generation for each feature
  :param evaluators:            Vector of functions that transform an individual parameters into costs (e.g. function that transform the name of the city to the kilometers of distance). First elem is the minimum, second is the maximum
  :param optimals:              Vector or optimal value for each feature (e.g. cost = 0, revenue = infinite)
  :param mutation_probability:  Value ([0,1]) used as probability of mutation
  :param population_size:       Integer defining the number of individuals generated in each generation

  
  Use case:
  
    >>> prototype = [[1,2,"red"]]
    >>> variables_type = ["numerical", "numerical", "categorical"]
    >>> ranges = [[0,1], [2,7], ["red", "blue", "green"]]
    >>> optimals = [0, 7, 0]
    >>> evaluators = [lambda x: 1-x**2, lambda x: x**2, lambda x: 0 if x == "red" else 1]
    >>> hall_of_fame_size = 10
    >>> number_iterations = 15
    >>> mutation_probability = 0.5
    >>> population_size=10000
    >>> one_population = EvolPop(prototype, variables_type, ranges, evaluators, optimals, mutation_probability, population_size)
    >>> one_population_best_individual = one_population.getBestIndividual()
  """  
  
  def __init__(self, prototype, variables_type, ranges, evaluators, optimals, mutation_probability = 0.5, population_size=10):
    """Initialize parameters"""    
    self.prototype=prototype
    self.variables_type=variables_type
    self.ranges=ranges
    self.evaluators=evaluators
    self.optimals=optimals
    self.mutation_probability=mutation_probability
    self.population_size=population_size
    self.population=[]
    
  def getBestIndividual(self):
    """Main function of the class. Generates a mutated population using the prototype as reference, and returns the best individual of the population. The individual is composed of two vectors [[paramaters],[distancesToOptimals]]"""    
    rdd = self.convertToRDD(prototype)
    return rdd\
              .flatMap(lambda x: self.generateMutatedPopulation(x))\
              .map(lambda x: self.individualAndDistance(x))\
              .reduce(lambda a,b: self.pairwiseSelection(a, b))

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
        
  def generateMutatedPopulation(self, initial):
    """Generates a population of individuals using the mutation probability and the ranges of the features"""
    assert len(initial) == len(self.ranges), "[generateMutatedPopulation] len(initial)=%d, len(self.ranges)=%d. They are not equal!" % (len(initial), len(self.ranges))
    assert len(initial) > 0, "[generateMutatedPopulation] len(initial) is not > 0, is %d" % len(initial)
    for j in range(self.population_size):
      new_individual = []
      for i in range(len(initial)):
        if SystemRandom().random() <= self.mutation_probability:
          if variables_type[i] == "numerical":
            # lineal between lower_range and upper_range
            number_inside_range = (float(self.ranges[i][1]) - float(self.ranges[i][0])) * SystemRandom().random() 
            new_individual.append(number_inside_range)
          elif variables_type[i] == "categorical":
            # random choice inside the options
            class_inside_range = SystemRandom().choice(self.ranges[i])
            new_individual.append(class_inside_range)
          else:
            raise TypeError("variables_type[",i,"] is not numerical nor categorical")
        else:
          new_individual.append(initial[i])
      self.population.append(new_individual)
    return self.population
    
  def pairwiseSelection(self, individual_and_distance_1, individual_and_distance_2):
    """Compares two individuals using pareto dominance and returns the most dominant one (the best of the two)"""
    assert len(individual_and_distance_1) == len(individual_and_distance_2) 
    individual_1_dominance = 0
    individual_2_dominance = 0
    for i in range(len(individual_and_distance_1[1])):
      if individual_and_distance_1[1][i] < individual_and_distance_2[1][i]:
        individual_1_dominance+=1
      elif individual_and_distance_1[1][i] > individual_and_distance_2[1][i]:
        individual_2_dominance+=1
    if individual_1_dominance >  individual_2_dominance:
      return individual_and_distance_1
    elif individual_1_dominance == individual_2_dominance:
      return individual_and_distance_1 # To do: solve tie break
    elif individual_1_dominance < individual_2_dominance:
      return individual_and_distance_2
    else:
      raise TypeError("pairwiseSelection failed!")

  def functionEvaluation(self, individual):
    """Transforms the individual parameters into costs using the evaluators"""
    assert len(individual) == len(self.evaluators)
    costs = []
    for i in range(len(individual)):
      costs.append(self.evaluators[i](individual[i]) )
    return costs

  def distancesToOptimals(self, costs):
    """Calculates the distances between the costs and the optimal costs (euclidean)"""
    assert len(costs) == len(self.optimals)
    return  [norm(c - o) for c,o in zip(costs, self.optimals)]

  def individualAndDistance(self, individual):
    """Appends the distances to individuals"""
    assert len(individual) == len(self.optimals)
    # transform into list to comply with reduce function expectations
    return [individual, self.distancesToOptimals(self.functionEvaluation(individual))] 

