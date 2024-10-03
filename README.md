
Here, I implement an experiment proposed by Paul Christiano [here](https://www.alignmentforum.org/posts/BxersHYN2qcFoonwg/experimentally-evaluating-whether-honesty-generalizes?commentId=dsDA2BWpHPdgLvaXX) to learn something about the generalization of transformers.
For simplicity I focus on a simple synthetic task: shortest
paths.

**the below document is not quite an accurate representation of
what I actually ended up doing. TODO: clean this up, and add some
documentation to the project**

# PLAN:

Let N be the maximum number of vertices in any graph that we ever consider.
Let D be a number such that most graphs that we consider have diameter at most D.

ARCH:
Let's stack D transformers.
To start, we are fed in an edge list.
Then we embed these and do transformer things.

Then, one way I could imagine performing the task is, in the i-th
layer you can compute whether or not you are distance i from
vertex 1. Or even closer. 
I haven't thought about exactly how you wire the self-attention +
residual connections etc to make this happen, but it seems
do-able.

Anyways, our training regiment has two steps
1. Train the network to compute shortest paths between vtx 1 and vtx 2 on Erdos-Renyi random graphs with number of vertices between 10 and 100 vertices.
2. Fine tune the network to compute shortest paths between vtx 1
   and vtx i for every other i, on Erdos-Renyi random graphs with
   number of vertices being between 10 and 20.

Then for evaluation we see 
1. How well does the model do at d(1,2)?
2. How well does the model do at d(1,i) in the small number of
   vertices regime?
3. Does the model generalize to handle d(1,i) in the large number
   of vertices regime?

# notes

Recall how a transformer works:

score(i,j) = Key[i] * Query[j]
alpha(i,j) = softmax(scores)
embedding(i) = sum_{j} alpha(i,j) Val[j]

Then we have a fully connected NN.
Next we do a layernorm.
After that we have a residual connection.

