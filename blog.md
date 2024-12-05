---
build: pandoc blog.md --citeproc --katex -s -o index.html
mkzip: zip project.zip index.html *.png
title: "6.7960 Project: Investigating Off-Distribution Generalization of Transformers"
bibliography: blog.bib
link-citations: true
---

<!-- Guidelines: https://www.dropbox.com/scl/fi/bet8enscln8ue36kd8t17/final_project_guidelines.pdf?rlkey=knd19cnumk51ho1y9crno56ib&e=2&dl=0 -->

<div style="text-align:center">
Anthony Wang, Alek Westover, Kevin Zhao

{xy,alekw,kevinmz}\@mit.edu
</div>

## Goals

Recently, LLMs have been developing very fast, and with that comes the concern of aligning the models to output true and productive statements. One common approach for ensuring this is to have a human in the loop rewarding the model for true outputs (e.g. RLHF), but one drawback to this problem is that humans can be poor judges of truthfulness. As LLMs become more capable, there might not even exist experts that are good judges of whether the model's outputs, such as difficult mathematical proofs, are truthful. So, we'd like to propose a potential solution to this issue via **off-distribution generalization** - applying human-like intuition to solve problems not in the dataset. Paul Christiano [proposed an experiment](https://www.alignmentforum.org/posts/BxersHYN2qcFoonwg/experimentally-evaluating-whether-honesty-generalizes?commentId=dsDA2BWpHPdgLvaXX) about shortest paths in a graph; our project is essentially to implement Christiano's proposed experiment. To the best of our knowledge, although there has been research in applying machine learning for different variations of graph searches [@10.5555/3666122.3666260], no one has done our exact experiment yet.

It is generally desirable for LLMs to output true statements. A current approach for ensuring this is to have a human in the loop rewarding the model for true outputs (e.g. RLHF); however, humans can be poor judges of truthfulness. We enjoy many cognitive biases and might employ superficial heuristics when judging truthfulness. A further challenge is that as LLMs develop further, there might not even exist experts that can correctly judge the accuracy and truthfulness of sophisticated outputs such as difficult mathematical proofs.

One approach to solving this problem is to reward an LLM for truthful behavior on simple inputs, and then hoping that the LLM generalizes its truthful behavior for more complex inputs where humans cannot provide helpful labels. Deep learning models often perform remarkable feats of off-distribution generalization -- for instance, a model trained to transform hand drawn cats into images of cats might be able to handle a "cat" with three eyes in an intuitive way. We might hope that generalizing truthfully is simple, thus promoted by "Occam's Razor", and aim to investigate that with this project.

COMMENT FROM KEVIN -- synthesize from intorduction

### Task

We will use a synthetic task to test our hypothesis that models will generalize truthfully off-distribution. The synthetic task is computing the distance between various vertices in an input graph. Our experiment will have three parts:

1. Pre-train a transformer to predict the distance between two fixed vertices $s,t$ on graphs with $n\in [8, 32)$ vertices.
2. Fine-tune a transformer to predict the distances between $s,t'$ for any $t'$ which is on the shortest path from $s$ to $t$, but only do fine-tuning on graphs with $n\in [8,16)$ vertices.
3. Test whether the transformer can accurately predict the distances between $s,t'$ for any $t'$ on the shortest path from $s$ to $t$ for graphs with $n\in [16,32)$ vertices.

### Related Work

COMMENT FROM ALEK 
-- please remove all mentions of graph neural networks -- that is BS: there is no actual reason why you'd ever use a Neural network to solve shortest paths, the point of choosing a synthetic task is because there is a **simple ground truth** which makes it easy to evaluate whether or not our model is performing correctly. We'd also hoped that the simplicity of the task would make it more feasible to do with a limited compute budget, but apparently this task was too hard for our architecture.


There has been some research into the algorithmic optimization of GNNs and how they may solve real-world issues; however, none of the related work targets using generic machine learning methods to solve graph problems.

- Cappart et al. has researched more into the Combinatorial Optimization of GNNs and developed algorithms for related tasks, thus facilitating machine learning [@DBLP:journals/corr/abs-2102-09544]. Their results are mostly algorithmic so we develop further by trading a bit of accuracy for much faster computation in such tasks.

- Tutsoy uses a graph-theory-based approach to model the epidemiological characteristics of infectious diseases, such as COVID-19 [@10.1109/TPAMI.2023.3256421]. We understand from his paper how GNN optimization may also be useful in researching novel diseases.

## Methods

### Algorithm for Shortest Paths

The standard algorithm to find the shortest path in a graph between a source numbered as $u$ and sink numbered as $v$ is **breadth-first search (BFS)**. The BFS algorithm maintains a mapping of visited vertices to their distances with respect to $u$, and each run of the algorithm goes through all the vertices newly visited in the previous run, and for each vertex, visits any of its unvisited neighbors. The algorithm terminates once either $v$ is visited or the set of newly visited vertices in a single run is empty.

We will use this algorithm to verify the accuracy of our machine learning approach. Given $V$ vertices and $E$ edges, the runtime of this algorithm is thus $O(V + E)$; however, a machine learning approach may do better in time through parallelism, although at the expense of using much more memory.

### Data

We will represent an $n$ vertex, $m$ edge unweighted, undirected graph as sequence of the endpoints of the $m$ edges, so $[a_1,b_1,a_2,b_2,\ldots,a_m,b_m]$ represents a graph with the edges $\{(a_i,b_i)\}$ for $1 \leq i \leq m$. We will pad all sequences to be the same length using the padding token 0.

The full input to our model will additionally add the target vertex after the padding tokens. The model is tasked with predicting the length of the shortest path between vertex 1 and the target vertex $t$. If no such path exists, we define the length to be $n+1$ which represents infinity. For example, an input-output pair for our model could look like $[1, 3, 3, 2, 0, 0, 0, 0, 2]$ and $2$ respectively.

We have three separate datasets.

- **Pre-train data**: For each $n \in [8,32)$, we will generate several graphs on $n$ vertices. We generate these graphs by inserting $2n$ random edges into the graph. We always set the target vertex to be $2$ here.
- **Fine-tune data**: For each $n \in [8,16)$, we will generate several graphs on $n$ vertices. We generate these graphs by inserting $2n$ random edges into the graph. We select the target vertex to be a random vertex on the shortest path from $1$ to $2$.
- **Generalization testing data**: The same as the fine-tune data, except we sample $n \in [16,32)$ instead.

We wrote some Python code to generate the data during the training loop, but Python is slow and the data generation wasted a lot of time during training. To get around this, we pre-generated the data before training and made our Python code multithreaded to speed it up.

### Architecture

TODO: honestly not much to say here since it's a pretty typical arch

We plan to use a standard transformer architecture. We will ensure that the number of layers in our transformer is at least the diameter of the graph. By doing this, we ensure that there is an extremely simple circuit --- namely BFS --- that the transformer could in theory learn to perform the task. Note that if the transformer actually learns a simple circuit to perform this task, then it seems more likely to generalize well. This is also our intuition for why it should be possible to fine tune on a small amount of data for finding shortest paths to other vertices besides $2$ -- it seems like the model should be computing these other distances as intermediate values in its computation to find the distance to vertex $2$.

### Embeddings

Since the order of the edges in the input does not matter, we did not use positional encodings. Each edge $(u,v)$ where $u < v$ is embedded to a dimension of $d$ where the first $\frac{d}{2}$ elements are the learned embedding of $u$ and the last $\frac{d}{2}$ elements are the learned embedding of $v$. For the target vertex $t$, we also embedded to dimension $d$, where the first $\frac{d}{2}$ elements are the learned embedding of $t$ and the last $\frac{d}{2}$ are a learned embedding of a special token.

## Training

For our model, we used a model dimension of 64, four layers, and two heads per layer, for a total of 200545 parameters in bfloat16 which corresponds to around 3.2e6 bits. The number of possible graphs on 15 vertices generated using our procedure is approximately

$$\frac{\binom{15}{2}^{15}}{15!} = 1.59\cdot10^{18}.$$

This is because there are $\binom{15}{2}$ choices for each of the 15 edges and we don't care about the order of the edges. This is only an approximation because some edges might be duplicated. Each graph has an answer between 1 and 15 which requires around 4 bits, so memorizing all the answers requires $4\cdot1.59\cdot10^{18} = 6.36\cdot10^{18}$, which is $2\cdot10^{12}$ times larger than our model size.



We used MSE loss, the Adam optimizer, a learning rate of 8e-4, and a batch size of 131072 for 8000 unique randomly generated batches. Our final MSE loss was approximately 0.3555.

![](training-loss.png)

![](training-2d-histogram.png)

One pattern we consistently noticed during training is that the model often gets stuck and plateaus for many epochs before rapidly decreasing. For instance, this happened between epochs 100 and 300 in the graph above:

![](grokking.png)

"grokking" hypothesis: it's memorizing all length 2 paths?

TODO: training curves for 1, 2, 3 length paths

### Potential Mathematical Approaches to Shortest Paths? Delete this?

Another way one can think of the shortest path of a graph is using a *matrix* to record which vertices are connected. Given vertices numbered $1$ to $V$, we denote the **adjacency matrix** $\textbf{M}$ of dimensions $V \times V$ as the matrix with element $\textbf{M}_{i, j} = 1$ if vertices $i$ and $j$ are connected by an edge and $\textbf{M}_{i, j} = 0$ if they are not. Now, we note that (1) For all $k$, $(\textbf{M}+I)^k_{i, j} = 0$ if and only if there exists no path from the vertex numbered $i$ to the vertex numbered $j$ that is distance $k$ or less due to Markov matrix processes. As a result, if the distance between vertices numbered $i$ and $j$ is $d$, then $\text{min}\left((\textbf{M}+I)^k_{i, j}, 1\right) = 1$ if $k \ge d$ and $\text{min}\left((\textbf{M}+I)^k_{i, j}, 1\right) = 0$ if $k < d$. 

With this information, because the distance between any two vertices is at most $V-1$ in a graph with $V$ vertices, we note that the *distance* matrix turns out to be simply $$\textbf{D} = \textbf{1}_{V \times V} \cdot V - \Sigma_{i=0}^{V-1}\text{min}\left((\textbf{M}+I)^k_{i, j}, 1\right).$$ The runtime to compute this is $O(V)$, although it will take more space to compute all powers of $\textbf{M}$.

## Fine tuning results

After receiving our initial results, we fine-tuned with a learning rate of 1e-5, also with MSE and the same batch size. Our final results are shown in the images below.

![](fine-tuning-loss.png)

![](fine-tuning-2d-histogram.png)

![](test-2d-histogram.png)

Memorization? Do some math here to compute how many bits required to memorize 1, 2, 3

## Complicated explicit transformer formula for shortest paths

```py
# Configuration
NVTXS = 16
MAXDIST = NVTXS + 1
AVGDEG = 2
SEQLEN = NVTXS + 1
HIDDENDIM = 4 * NVTXS + 2

# Start indices for different sections of the input data
START_REACH = NVTXS + 1
START_OUT = 2 * NVTXS + 1
START_SELF = 3 * NVTXS + 1
SRC_FLAG_IDX = START_SELF
ANS_FLAG_IDX = 0
NOTANS_FLAG_IDX = -1

BIG = 20
SUPABIG = 100
MED = 10
CURSE = 5

class SillyTransformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        with torch.no_grad():
            # Initialize weight parameters with specific configurations
            self.mostKs = nn.ParameterList()
            self.mostQs = nn.ParameterList()
            self.mostVs = nn.ParameterList()
            for head in range(1, NVTXS + 1):
                Q = nn.Parameter(torch.zeros((2, HIDDENDIM), device=device))
                Q[0, START_REACH - 1 + head] = SUPABIG
                Q[1, NOTANS_FLAG_IDX] = 1

                K = nn.Parameter(torch.zeros((2, HIDDENDIM), device=device))
                K[0, head] = 1
                K[1, ANS_FLAG_IDX] = BIG

                V = nn.Parameter(torch.zeros((NVTXS, HIDDENDIM), device=device))
                for i in range(NVTXS):
                    V[i, START_SELF + i] = 1

                self.mostKs.append(K)
                self.mostQs.append(Q)
                self.mostVs.append(V)

            self.weirdKs = nn.ParameterList()
            self.weirdQs = nn.ParameterList()
            self.weirdVs = nn.ParameterList()
            for layer in range(NVTXS):
                K = nn.Parameter(torch.zeros((3, HIDDENDIM), device=device))
                K[0, NOTANS_FLAG_IDX] = -BIG
                K[0, SRC_FLAG_IDX] = BIG+SUPABIG
                K[1, NOTANS_FLAG_IDX] = -SUPABIG
                K[1, NVTXS + 2] = BIG+SUPABIG
                K[1, ANS_FLAG_IDX] = -BIG-SUPABIG
                K[2, ANS_FLAG_IDX] = MED

                Q = nn.Parameter(torch.zeros((3, HIDDENDIM), device=device))
                Q[:, ANS_FLAG_IDX] = 1

                V = nn.Parameter(torch.zeros((NVTXS, HIDDENDIM), device=device))
                V[layer, SRC_FLAG_IDX] = 1

                self.weirdKs.append(K)
                self.weirdQs.append(Q)
                self.weirdVs.append(V)

    def forward(self, src):
        for layer in range(NVTXS):
            allKs = [self.weirdKs[layer]] + [x for x in self.mostKs]
            allQs = [self.weirdQs[layer]] + [x for x in self.mostQs]
            allVs = [self.weirdVs[layer]] + [x for x in self.mostVs]
            head_outputs = []
            
            for (K, Q, V) in zip(allKs, allQs, allVs):
                ksrc = torch.matmul(src, K.unsqueeze(0).transpose(-2, -1))
                qsrc = torch.matmul(src, Q.unsqueeze(0).transpose(-2, -1))
                vsrc = torch.matmul(src, V.unsqueeze(0).transpose(-2, -1))

                scores = torch.matmul(qsrc, ksrc.transpose(-2, -1))
                attention_weights = torch.softmax(scores, dim=-1)
                head_output = torch.matmul(attention_weights, vsrc)
                head_outputs.append(head_output)

            new_reaches = sum(head_outputs[1:])
            BSZ = new_reaches.shape[0]

            nodelta_nbrs = torch.zeros((BSZ, SEQLEN, NVTXS + 1), device=self.device)
            morepadlol = torch.zeros((BSZ, SEQLEN, 1 + NVTXS), device=self.device)

            src = src + torch.cat((nodelta_nbrs, new_reaches, head_outputs[0], morepadlol), dim=2)
            src[:, :, START_REACH:START_REACH + NVTXS] = 2 * torch.sigmoid(src[:, :, START_REACH:START_REACH + NVTXS] * CURSE) - 1

        canreach = src[:, 0, START_OUT:START_OUT + NVTXS]
        final_output = 1 + torch.sum(1 - canreach, dim=1)
        return final_output
```

It looked like the fine tuning results weren't as meaningful because TODO: ALEK

## Customizing a Transformer

After much deliberation, we decided the next step for us was to customize a transformer, writing it ourselves. We observed that we wished for the transformer to do similar work as a BFS. As a result, we decided to work with the following transformer, for a graph with $n$ vertices $v_1, v_2, \cdots, v_n$:

\begin{array}{|c|c|c|c|c|c}
\text{ANS} & v_{1} & v_{2} & \cdots & v_{n} & \\ \hline
1 & 0 & 0 & \cdots & 0 & \text{ANS}\\ \hline
\text{ANS} & \text{NBR}_{1} & \text{NBR}_{2} & \cdots & \text{NBR}_{n} & \text{NBR}\\ \hline
\text{ANS} & \text{REACH}_{1} & \text{REACH}_{2} & \cdots & \text{REACH}_{n} & \text{REACH}\\ \hline
\text{ANS} & \text{SELF}_{1} & \text{SELF}_{2} & \cdots & \text{SELF}_{n} & \text{SELF}\\ \hline
V_{\text{OUT}} & NULL& NULL& NULL& NULL& \text{OUT}\\ \hline
0 & 1 & 1 & \cdots &1 & \text{NOT}\\ \hline
\end{array}

Specifically, we see that $\text{NBR}_{i}$ is a $n \times 1$ vector detailing which of the vertices are neighboring vertex $v_i$, so the $j$th element of $v_i$ is $1$ if $v_i$ and $v_j$ are neighboring vertices, and $0$ otherwise. Additionally, $\text{SELF}_{i}$ is just the $n \times 1$ vector with the $i$th element $1$ and all other elements $0$ (e.g. the one-hot encoding of the vector). Now, at every step, the $\text{REACH}_k$ vector for all $k$ is updated based on the previous $\text{REACH}_k$ vector and $\text{NBR}_{k}$ (since all entries that are $1$ in $\text{REACH}_k\text{NBR}_{k}^T$ must be updated in the manner such that if the $(i, j)$th element of $\text{REACH}_k\text{NBR}_{k}^T$ is $1$, then $\text{REACH}_i$'s $j$th column is set to $1$. This is equivalent to adding $\text{REACH}_k$ to each integer $i$ where  $\text{NBR}_{k}$'s $i$th entry is nonzero.

This iterates through all the vertices, and at the end, we may see what run we are on to update $V_{\text{OUT}}$.

## Conclusion

just do bfs lol

## References
