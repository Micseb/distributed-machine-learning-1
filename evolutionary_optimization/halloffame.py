"""
Author: Santiago Morante
DISTRIBUTED MULTIOBJECTIVE OPTIMIZATION BASED ON EVOLUTIVE ALGORITHMS
"""


###################################################################################
###########################   CLASS HALL OF FAME  ################################
###################################################################################
from random import SystemRandom

class HallOfFame():
  """
  HALLOFFAME:
  Contains the best individuals of each population. Can accept or reject new individuals if they are better than any of its individuals. It also cross individualas the obtain new crossover prototype to generate new populations
    
  :param size:   Number of best individuals to keep at maximum
  
  Use case:
    >>> hall_of_fame = HallOfFame(size=10)
    # Until hall of fame is full
    >>> best_individual = [generate a population and obtain best individual]
    >>> hall_of_fame.addElement(best_individual)
    # Once the hall of fame is full
    >>> best_individual = [generate a population and obtain best individual]
    >>> hall_of_fame.acceptOrReject(one_population_best_individual)
    # To obtain a descendant from two individuals
    >>> descendant = hall_of_fame.getCrossoverDescendant()
  """  
  
  def __init__(self, size=10):
    """Initialize parameters"""
    self.size=size
    self.ordered=False
    self.elements=[]
      
  def addElement(self, new_element):
    """Adds element to the hall of fame until it is full. Does nothing instead"""
    if len(self.elements) < self.size:
      self.elements.append(new_element)
    else:
      print "Maximum number of individuals reached in Hall of fame! Cannot add more!"
    
  def sortElements(self, reverse=True):
    """Sorts elements in hall of fame. Top=Worst, Bottom=Best. This can be overrided using reverse parameter"""
    self.elements = sorted(self.elements, cmp=lambda x,y: self.pairwiseSelectionNumber(x,y), reverse=True)
    self.ordered = True
  
  def getCrossoverDescendant(self):
    """Combines two individuals to produce a descendant. It randomly (prob=0.5) chooses a feature from one parent or another"""
    candidate_1 = SystemRandom().choice(self.elements)
    candidate_2 = SystemRandom().choice(self.elements)
    assert len(candidate_1[0]) == len(candidate_2[0])
    descendant = []
    for i in range(len(candidate_1[0])):
      if SystemRandom().random() > 0.5:
        descendant.append(candidate_1[0][i])
      else:
        descendant.append(candidate_2[0][i])
    # to converted into list elem        
    return [descendant] 

  def acceptOrReject(self, candidate):
    """Evaluates (using pareto dominance) a candidate individual with the elements of hall of fame. If it is better than one of them, the loser is replaced by the winner"""
    if self.ordered == False:
      self.sortElements()
    for i in range(self.size):
      if self.pairwiseSelectionNumber(candidate, self.elements[i]) == -1 and candidate != self.elements[i]:
        self.elements[i]=candidate
        break 

  def pairwiseSelectionNumber(self, individual_and_distance_1, individual_and_distance_2):
    """Compares two individuals using pareto dominance and returns a number indicating the most dominant one (the best of the two)"""    
    assert len(individual_and_distance_1) == len(individual_and_distance_2) 
    individual_1_dominance = 0
    individual_2_dominance = 0
    for i in range(len(individual_and_distance_1[1])):
      if individual_and_distance_1[1][i] < individual_and_distance_2[1][i]:
        individual_1_dominance+=1
      elif individual_and_distance_1[1][i] > individual_and_distance_2[1][i]:
        individual_2_dominance+=1
    if individual_1_dominance > individual_2_dominance:
      return -1
    elif individual_1_dominance == individual_2_dominance:
      return 0 
    elif individual_1_dominance < individual_2_dominance:
      return 1
    else:
      raise TypeError("pairwiseSelectionNumber failed!")

