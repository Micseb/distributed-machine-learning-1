# Distributed Multiobjective Evolutionary Optimization

In this repository you can find a distributed version of a multiobjective evolutionary optimization algorithm.

The classical paradigm of evolutionary optimization has been adapted to fit in the computational model of Spark. The algorithm generates big populations of individuals, evaluates them pairwise (using pareto dominance) and return the best individual. This process can be repeated many times. Each time a best individual is obtained, it is stored in a hall of fame of limited size. When this size is exceeded, a new candidate individual is compared (using pareto dominance) with the already stored ones. If better, the loser is replaced by the winner.

Due to the way the individuals are compared using pareto dominance, this algorithm can be used for multiobjective optimization.

This repository includes two classes:

1. **EvolPop**: This class allows to generate a population of individuals mutated from a provided prototype. After the population is generated, a pairwise selection is performed to obtain the best individual.

2. **HallOfFame**: This class implements the storage place for the best individuals of the population. It also generates crossover descendant of its stored individuals if asked.


The classical three steps of evolutionary algorithm are performed in the following way:

* **Mutation**: A population of individuals, mutated from a prototype, is generated.

```
#in EvolPop
...flatMap(lambda x: self.generateMutatedPopulation(x))
```

* **Selection**: All the individuals of the population are evaluated to obtain the best one.

```
#in EvolPop
...reduce(lambda a,b: self.pairwiseSelection(a, b))
```
* **Crossover**: From the hall of fame, two individuals are selected randomly and uniformly crossovered. The descendant can be used to generated a new mutated population.

```
#in HallOfFame
hall_of_fame.getCrossoverDescendant()
```
	
A notebook is included with an example of use.