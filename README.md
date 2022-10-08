# heart-disease-prediction
A Bayesian belief network describes the probability distribution over a set of 
variables. 
Probability
P(A) is used to denote the probability of A. For example if A is discrete with states {True, False} then 
P(A) might equal [0.2, 0.8]. I.e. 20% chance of being True, 80% chance of being False.
Joint probability
A joint probability refers to the probability of more than one variable occurring together, such as 
the probability of A and B, denoted P(A,B).
Conditional probability
Conditional probability is the probability of a variable (or set of variables) given another variable (or 
set of variables), denoted P(A|B).For example, the probability of Windy being True, given that Raining 
is True might equal 50%.This would be denoted P(Windy = True | Raining = True) = 50%.
Once the structure has been defined (i.e. nodes and links), a Bayesian network requires a probability 
distribution to be assigned to each node.Each node X in a Bayesian network requires a probability 
distribution P(X | pa(X)).Note that if a node X has no parents pa(X) is empty, and the required 
distribution is just P(X) sometimes referred to as the prior.This is the probability of itself given its 
parent nodes.
If U = {A1,...,An} is the universe of variables (all the variables) in a Bayesian network, and pa(Ai) are 
the parents of Ai then the joint probability distribution P(U) is the simply the product of all the 
probability distributions (prior and conditional) in the network, as shown in the equation below.This 
equation is known as the chain rule.
From the joint distribution over U we can in turn calculate any query we are interested in (with or 
without evidence set).
Suppose that there are two events which could cause grass to be wet: either the sprinkler is on or it's 
raining. Also, suppose that the rain has a direct effect on the use of the sprinkler (namely that when it
rains, the sprinkler is usually not turned on). Then the situation can be modeled with a Bayesian 
network (shown to the right). All three variables have two possible values, T (for true) and F (for 
false)
The joint probability function is:
The model can answer questions like "What is the probability that it is raining, given the grass is wet?"
 byusing the conditional probability formula and summing over all nuisance variables:
 Heart Disease Databases 
The Cleveland database contains 76 attributes, but all published experiments refer to using a subset of 14 of 
them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. 
The "Heartdisease" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no 
presence) to 4. 
Database : 0 1 2 3 4 Total 
Cleveland: 164 55 36 35 13 303
Attribute Information:
![image](https://user-images.githubusercontent.com/92110686/194722867-2c753c7f-01ad-499d-b6c3-8ff22560ce8b.png)
![image](https://user-images.githubusercontent.com/92110686/194722883-92d75c9a-5664-4c17-a884-6a50e8699fbc.png)
![image](https://user-images.githubusercontent.com/92110686/194722899-cc2168ac-774b-4878-a5ee-4996bb614748.png)
![image](https://user-images.githubusercontent.com/92110686/194722905-c37eb720-80ac-4dbd-b2d7-df84ac0c8151.png)
![image](https://user-images.githubusercontent.com/92110686/194722895-75a364ab-4126-45b0-822a-19144b098f96.png)
