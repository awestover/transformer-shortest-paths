1. Submit proposal [10 of grade] (Due: November 14, 11:59pm): Submit a pro- posal as a one page pdf. Provide an outline of your plan for the project and questions you will investigate / analysis you’ll conduct in the course of it. It may help to define a set of hypotheses you will test. An integral aspect of the proposal is to define a project idea that is both realistic and ambitious in scope. We recommend that you use the project proposal stage to get feedback from the teaching staff on the project’s feasibility and whether the proposal satisfies the project expectations of the class. 


Specify architecture stuff

Specify the training data generation process

undirected graph

[XY == write out how we're gonna generate data]
PRE-train data 

Fine-tune data

validation data

- hypothesis 1 -- transformers can learn shortest paths without too much GPUs

mathemetical motivation for why this is possible with a not super deep transfomer. 

- hypothesis 2 -- pre-training on 1-2 shortest path should make fine-tuning for other shortest paths which are prefix of the shortest 1-2 path faster

we believe this because the info should be sitting somewhere inside the model

- hypothesis 3 -- training for lots of sizes of 1-2  paths, and fine tuning on small graphs, it'll generalize to large graphs.

we hope that this is like Occam's razor 

train on erdos renyi graphs, does it generalize to arbitrary graphs?

Inspiration for project 
Here, I implement an experiment proposed by Paul Christiano [here](https://www.alignmentforum.org/posts/BxersHYN2qcFoonwg/experimentally-evaluating-whether-honesty-generalizes?commentId=dsDA2BWpHPdgLvaXX) 
