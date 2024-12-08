---
build: pandoc blog.md --citeproc --katex -s -o index.html
mkzip: zip project.zip index.html *.png
title: "Discovering Graph Algorithms Using Transformers"
bibliography: blog.bib
link-citations: true
---

<!-- Guidelines: https://www.dropbox.com/scl/fi/bet8enscln8ue36kd8t17/final_project_guidelines.pdf?rlkey=knd19cnumk51ho1y9crno56ib&e=2&dl=0 -->

<div style="text-align:center">
Anthony Wang, Alek Westover, Kevin Zhao

{xy,alekw,kevinmz}\@mit.edu
</div>

## Motivation

Transformers--the architecture that powers LLMs--can do incredible feats: trained on hundreds of gigabytes of raw text, they can learn to hold natural conversations, reason about the physical world, and write code. Skeptics argue that LLMs are simply memorizing their datasets without gaining any deeper understanding. For instance, GPT's o1 model, achieving 90th percentile on Codeforces, struggles with simple but bizarre algorithms problems such as "find the subarray of a 2D array with the minimum average". In this project, we hope to explore **when off-distribution generalization happens in a transformer**. Paul Christiano proposed an experiment [here](https://www.alignmentforum.org/posts/BxersHYN2qcFoonwg/experimentally-evaluating-whether-honesty-generalizes?commentId=dsDA2BWpHPdgLvaXX) about shortest paths in a graph to investigate this, so we decided to become the first to implement his experiment and put transformers' generalization abilities to the test.

LLMs are notorious for making up complete nonsense, so we also hope that our project can shed light on when truthfulness generalizes. It's generally desirable for LLMs to output true statements. One current approach for ensuring this is to have a human in the loop rewarding the model for true outputs (e.g. RLHF). However, humans can be poor judges of truthfulness and have many cognitive biases and superficial heuristics. A further challenge is that as LLMs become more capable, there might not even exist experts that are good judges of whether the models outputs, such as difficult mathematical proofs, are truthful. For instance, most Task Rabbit workers would probably be hard pressed to evaluate whether a difficult mathematical proof produced by an LLM is true. The entire mathematical community has been known on occasion to [believe false statements for many years](https://en.wikipedia.org/wiki/Grunwald%E2%80%93Wang_theorem).

One possible solution is to reward an LLM for truthful behavior on simple inputs, and then hope that the LLM generalizes its truthful behavior for more complex inputs where humans cannot provide helpful labels. Deep learning models can be remarkably good at off-distribution generalization--for instance, a model trained to transform hand drawn cats into images of cats might be able to handle a "cat" with three eyes in an intuitive way. We might hope that generalizing truthfully is simple, thus promoted by "Occam's Razor".

## Related Work

COMMENT FROM ALEK 
-- please remove all mentions of graph neural networks -- that is BS: there is no actual reason why you'd ever use a Neural network to solve shortest paths, the point of choosing a synthetic task is because there is a **simple ground truth** which makes it easy to evaluate whether or not our model is performing correctly. We'd also hoped that the simplicity of the task would make it more feasible to do with a limited compute budget, but apparently this task was too hard for our architecture.


There has been some research into the algorithmic optimization of GNNs and how they may solve real-world issues; however, none of the related work targets using generic machine learning methods to solve graph problems.

- Cappart et al. has researched more into the Combinatorial Optimization of GNNs and developed algorithms for related tasks, thus facilitating machine learning [@DBLP:journals/corr/abs-2102-09544]. Their results are mostly algorithmic so we develop further by trading a bit of accuracy for much faster computation in such tasks.

- Tutsoy uses a graph-theory-based approach to model the epidemiological characteristics of infectious diseases, such as COVID-19 [@10.1109/TPAMI.2023.3256421]. We understand from his paper how GNN optimization may also be useful in researching novel diseases.



## Task

Our synthetic task is simple: compute the distance between various vertices in an input graph. To test off-distribution generalization, our experiment has three steps.

1. **Pre-train** a transformer to predict the distance between vertices $1$ and $2$ in graphs with $n \in [3,15]$ vertices.

<div style="text-align:center">
![](img/train.svg)
</div>

2. **Fine-tune** a transformer to predict the distances between vertex $1$ to $t$ for any $t$ on the shortest path from $1$ to $2$, but only do fine-tuning on graphs with $n \in [3,7]$ vertices.

<div style="text-align:center">
![](img/finetune.svg)
</div>

3. **Test** whether the transformer can accurately predict the distances between $1$ to $t$ for any $t \leq 7$ on the shortest path from $1$ to $2$ for graphs with $n \in [3,15]$ vertices.

<div style="text-align:center">
![](img/test.svg)
</div>

### Algorithm for Shortest Paths

The standard algorithm to find the shortest path in a graph between vertices $u$ and $v$ is **breadth-first search (BFS)**, taught in every intro algorithms class. Initially, BFS starts at $u$, and at each phase, explores a farther layer of vertices from $u$. During a phase, BFS goes through every vertex in the current layer and adds any of their unvisited neighbors to the next layer. The algorithm terminates once we reach $v$ or if the next layer is empty. For a graph with $V$ vertices and $E$ edges, the runtime of BFS is $O(V + E)$. BFS gives us an easy and fast way to find the ground truth answer for any graph, so that we can verify the accuracy of our machine learning approach.

We hope that our model can learn BFS or some other simple, generalizable algorithm for shortest paths, because the model can't just pull some magic number out of a hat but intuitively needs to count or list the vertices on the shortest path from $1$ to $2$. In fact, we will show how to hand-craft a set of weights to implement BFS in a transformer, so it's indeed theoretically possible for a transformer to achieve 100% accuracy.

### Data

We'll represent an $n$ vertex, $m$ edge unweighted, undirected graph as sequence of the endpoints of the $m$ edges, so $[a_1,b_1,a_2,b_2,\ldots,a_m,b_m]$ represents a graph with the edges $\{(a_i,b_i)\}$ where $a_i < b_i$ for $1 \leq i \leq m$. All sequences are padded to the same length using the padding token $0$.

The full input to our model additionally includes the target vertex $t$ after the padding tokens. The label to an input is the length of the shortest path from $1$ to $t$. If no such path exists, we define the length to be $n+1$ which represents infinity. For example, the input $[1, 3, 3, 4, 2, 4, 2, 3, 0, 0, 0, 0, 3]$ has the label $2$.

<div style="text-align:center">
![](img/finetune.svg)
</div>

We have three datasets for each step.

1. **Pre-train data**: For each $n \in [3,15]$, we generated an equal number of graphs on $n$ vertices, with $t = 2$. Each graph was created by choosing $n$ random edges.
2. **Fine-tune data**: For each $n \in [3,7]$, we generated an equal number of graphs on $n$ vertices each with a random $t$ on the shortest path from $1$ to $2$. Again, each graph was created by choosing $n$ random edges.
3. **Generalization test data**: The same as the fine-tune data, except we sample $n \in [3,15]$ and $t \leq 7$.

We wrote some Python code to generate the data during the training loop, but Python is excruciatingly slow and data generation wasted a lot of training time. Our solution was to pre-generate the data before training using a multithreaded version of our Python code.

## Complicated explicit transformer formula for shortest paths

TODO: Kevin or Alek

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

## Perturbing the Weights

SLT folks like to think about geometry of loss landscape CITE
So we did an experiment where we mess with the weights. 

Findings: XXX


## Our Model

### Architecture

We used a standard transformer architecture. To ensure that it can in theory learn BFS, we ensured that the number of layers in our transformer exceeds the diameter of the input graphs.

Since the order of the edges in the input doesn't matter, we did not use positional encodings. Each edge $(a,b)$ is embedded to dimension $d$ where the first $\frac{d}{2}$ elements are the learned embedding of $a$ and the last $\frac{d}{2}$ elements are the learned embedding of $b$. For the target vertex $t$, we pair it with the special token $TARGET$ and embed $(t,TARGET)$ in the same way.

<!-- https://cocreate.csail.mit.edu/r/sxArTEXiAgJshznmm -->
![](img/embeddings.svg)

### Training

To match the BFS transformer as closely as possible, we used a model dimension of $64$, $11$ layers, and $2$ heads per layer, for a total of 550433 parameters. In 32-bit float precision, that corresponds to around $1.76\cdot10^6$ bits. The number of possible graphs on 15 vertices generated using our procedure is approximately

$$\frac{\binom{15}{2}^{15}}{15!} = 1.59\cdot10^{18}.$$

This is because there are $\binom{15}{2}$ choices for each of the 15 edges and we don't care about the order of the edges. This is only an approximation because some edges might be duplicated. Each graph has an answer between 1 and 15 which requires around 4 bits, so memorizing all the answers requires $4\cdot1.59\cdot10^{18} = 6.36\cdot10^{18}$ bits, which is $3.61\cdot10^{12}$ times larger than our model size.

To train the model, we used MSE loss, the Adam optimizer, a learning rate of $3\cdot10^{-4}$, and a batch size of $2^{15}$ for one billion randomly generated graphs. A training run takes roughly eight hours to run on a Radeon 7900 XTX graphics card. Our final MSE loss was $0.000555$.

TODO: use https://mpld3.github.io/index.html to make interactive plots

![](training-loss.png)

![](training-2d-histogram.png)

One pattern we consistently noticed during training is that the model often gets stuck and plateaus for many epochs before rapidly decreasing. For instance, this happened between epochs 100 and 300 in the graph above:

![](grokking.png)

"grokking" hypothesis: it's memorizing all length 2 paths?

TODO: cite Neel Nanda grokking modular addition

TODO: CRAZY!!! training curves for 1, 2, 3 length paths

One pitfall we encountered during training is that we initially used bfloat16 to save VRAM, but our huge batch size caused loss-of-precision problems and made training very difficult. It took us two weeks to debug this until we found that switching to float32 improved training significantly.

## Fine tuning results

After receiving our initial results, we fine-tuned with a learning rate of 1e-5, also with MSE and the same batch size. Our final results are shown in the images below.

![](fine-tuning-loss.png)

![](fine-tuning-2d-histogram.png)

![](test-2d-histogram.png)

TODO: get new graphs

It's pretty good!!!

Can only generalize to target vertices from 2 to 7 since 8 through 15 didn't appear in the fine-tune data

but this still means it

## Conclusion

however, a machine learning approach may do better in time through parallelism, although at the expense of using much more memory.
**TODO: ALEK: this is BS. If you want a parallel algorithm for BFS, here's one https://en.wikipedia.org/wiki/Parallel_single-source_shortest_path_algorithm**

just do bfs lol

**Future Work**
There are a couple of other things that we could try to learn shortest paths better and maybe see more generalization. 
- Chain of thought
- Train model to output a path, not just the distance. Give it partial points for outputting anything that is a legitimate path (i.e., consists of real edges) and more points for getting the distance correct. 

## References
