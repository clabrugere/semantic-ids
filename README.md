# Semantic IDs learned with RQ-VAE

Train a model mapping dense embeddings to sequences of codes $[c_0, ..., c_{K-1}]$, or semantic-ids, using residual quantization and export the resulting representation with collision resolution for consumption by a generative recommender.

Implements residual quantization by snapping each level's residual to one learned vector of the level's codebook, with dead-code revival and k-means codebook initialization. K levels are used for semantic codes, plus another one to disambiguate colliding prefixes at export time, to ensure each item gets a distinct code sequence.

## Usage

Refer to `example/train.py` to train the model and then generate code sequences:

```bash
uv run example/train.py --data my_data.parquet --out semantic_ids.pt
```

Where the data file must contain an embedding column.

## How It Works

In recommender systems, item space can sometimes have very large cardinality: for example a marketplace managing millions of products, or a social network with hundreds of millions of posts and videos. 

Mapping a unique vector embedding to each item then becomes impractical. Not only because that requires maintaining huge embedding tables that scale with the number of items, on device, but often the supervised signal used for learning is also very sparse and noisy at the individual item level. Semantic-ids have emerged as a way to address those issues by leveraging a hierarchical representation learned by iterative residual quantization. 

### Quantization and codebook learning

Concretely, it consists in learning an encoder that maps item embeddings (for instance content embeddings from a CLIP model) to a finite sequence of codes: $e \mapsto [c_0, ..., c_k, ..., c_{K-1}]$ where each level $k$ is an independent codebook of $V$ embeddings. With $K$ separate codebooks, we can represent up to $V^K$ semantic prefixes by only storing $K \cdot V$ embeddings (instead of the full catalog), and a static map of item IDs to code-sequences to perform the tokenization when working with the generative recommender.

Starting with a projected input embedding $z$ (through a simple MLP encoder), each level quantizes its input by snapping it to the nearest code $c$ in the level's codebook, before computing the residual $r = z - c$ and passing it to the next level.

Codebooks are optionally initialized with k-means centroids of each level's residuals to improve codebook utilization. Then the codebooks can be trained two ways:

1. Jointly with the encoder through the optimizer using a straight-through estimator (as arg min used to assign codes is not differentiable). Codebooks and the encoder are learned by minimizing the loss $L = \text{MSE}(x_{reconstructed}, x) + L_{VQ}$ where codebooks are learned through $L_{VQ} = K \cdot ( \text{MSE}(C, sg(R)) + \lambda \cdot \text{MSE}(R, sg(C)))$, $K$ being the number of levels, $C$ and $R$ the selected code vectors and the residuals across levels respectively and $sg(.)$ the stop gradient operator. The first term of $L_{VQ}$ pulls codes towards their assigned residuals while the second term regularizes the encoder to keep the residuals close to their selected code.

2. Using an exponential-moving average of the residuals assigned to a given code, decayed by some constant factor each batch. The loss returned is the same, though the first term of $L_{VQ}$ contributes no gradients and only the second term and the reconstruction contribute to learn the encoder.

### Dead code revival

Some codes can die if they are never assigned to a residual, degrading codebook utilization and hence expressivity and wasting space. To mitigate this, the model keeps usage statistics per code (decayed by some constant factor each batch). If a code's usage drops below a threshold, it is considered dead and can be revived by assigning it a random residual. Code usage and revival have to be explicitly called in the training loop, after the optimizer step.

### Codebook normalization during assignment step

By default, codebooks are L2 normalized for the assignment step, in order to neutralize the norm's magnitude influence. Reconstruction and the next level's residual use the raw, unnormalized code.

### Disambiguation codes

In practice, K levels might not be enough to exhaustively represent all the input embeddings: because of the quantization, some items can end up with the same code sequence, creating collisions. A final level that doesn't quantize a residual is used to resolve the collisions. Items sharing a prefix can be ranked by some key, such as  the residual norm, and assigned ordinal codes starting at 0, within that group.

Note that this is not necessarily stable between input changes.

### Decoding for candidate generation

Another interesting property of this method; and related to the fact that items are represented as a code sequence; is the emerging hierarchical structure: earlier codes can (but without any guarantees) capture coarse features of the items (like a category of a product) while the later ones, finer attributes (like the color of a shirt). This is particularly interesting as it can help with cold-start and supervised signal sparsity.

In generative recommenders, this approach requires a very specific decoding though. First, the number of decoding steps scales with $K$. Second, as there is a dependency between codes in an item's code sequence representation (code $k$ depends on the code $k-1$), we cannot naively decode over the whole codebook given the prefix (all previously decoded codes), otherwise we risk generating a code sequence for a nonexistent item. In practice, we can avoid that by constraining the decoding using an auxiliary trie-like data structure, masking invalid continuations (by simply overwriting logits with -inf on invalid codes to force the probability mass on valid codes only).

From this ensues a main drawback of the method: decision errors cannot be corrected as decoding progresses. If at some point the generative model samples the wrong code, it gets stuck on this branch till the end, and retrieves an irrelevant item. To mitigate this, a beam-search over multiple trie branches is used instead of decoding along a single branch. It starts with a beam with a predefined width $B$, and at each code level, those are expanded to their top $B$ valid codes, before globally pruning back to $B$ branches using the cumulative log-probability of the prefix.

When the last semantic level $K-1$ is reached, we end up with at most $B$ prefixes, ranked by cumulative log-probability (over the prefix). To handle the last disambiguation level $K$, a specific policy needs to be specified: for instance returning all the items in the collision group, random sampling, top-k using an auxiliary signal like conversions, etc...