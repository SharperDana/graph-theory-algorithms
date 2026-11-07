# Graph Theory & Compression Algorithms in Python

An algorithmic implementation featuring optimal prefix-free data compression (**Huffman Coding**) and unrooted **Tree Isomorphism & Canonical Vertex Mapping** (**AHU Algorithm**).

---

## Features & Algorithms

### 1. Optimal Binary Text Compression (Huffman Coding)
* **`encode_text(text)`**: Constructs an optimal prefix-free binary code tree bottom-up using a min-heap priority queue based on character frequencies.
  * Deterministic tie-breaking and canonical dictionary ordering (descending frequency, then lexicographical).
  * Handles edge cases seamlessly (empty inputs, single-character alphabets).
* **`decode_text(encoded_text, codes)`**: Reconstructs the original string in linear time using a prefix-tree (Trie) traversal.

### 2. Unrooted Tree Isomorphism & Bijection Mapping (AHU Algorithm)
* **`are_isomorphic(tree_1, tree_2)`**: Determines whether two undirected, unrooted trees (given as adjacency matrices) are structurally isomorphic.
* **`mapping_isomorphic(tree_1, tree_2)`**: Computes the exact bijective vertex mapping $f: V_1 \rightarrow V_2$ that preserves adjacency relations.
* **Algorithmic Approach**:
  1. **Center Finding**: Identifies the 1 or 2 Jordan centers of each tree via iterative leaf-peeling in $O(V)$ time.
  2. **Canonical Encoding (AHU)**: Roots the trees at their centers and computes canonical parenthesis string encodings bottom-up using iterative post-order traversal (avoiding recursion depth limits on degenerate trees).
  3. **Vertex Matching**: Pairs isomorphic subtrees in linear time by traversing sorted canonical children lists.

---

## Complexity Analysis

| Function | Time Complexity | Space Complexity | Notes |
| :--- | :--- | :--- | :--- |
| `encode_text` | $O(N + \Sigma \log \Sigma)$ | $O(\Sigma + M)$ | $N$ = text length, $\Sigma$ = unique chars, $M$ = encoded bit length |
| `decode_text` | $O(M + \Sigma \cdot L)$ | $O(\Sigma \cdot L)$ | $L$ = max codeword length |
| `are_isomorphic` | $O(V^2)$ input conversion / $O(V \log V)$ tree processing | $O(V)$ | $O(V^2)$ due to reading the $V \times V$ adjacency matrix |
| `mapping_isomorphic` | $O(V^2)$ input conversion / $O(V \log V)$ tree processing | $O(V)$ | Returns sorted vertex pairs `[(0, w_0), (1, w_1), ...]` |

---

## Usage Example

```python
from Algo_Graph_Project import encode_text, decode_text, are_isomorphic, mapping_isomorphic

# 1. Huffman Coding
text = "this is an example of a huffman tree"
encoded, codes = encode_text(text)
decoded = decode_text(encoded, codes)
assert text == decoded

# 2. Tree Isomorphism & Vertex Mapping
T1 = [
    [False, True,  True,  False],
    [True,  False, False, False],
    [True,  False, False, True],
    [False, False, True,  False]
]
T2 = [
    [False, True,  False, True],
    [True,  False, False, False],
    [False, False, False, True],
    [True,  False, True,  False]
]

if are_isomorphic(T1, T2):
    mapping = mapping_isomorphic(T1, T2)
    print("Vertex bijection:", mapping)
    # Output: [(0, 0), (1, 1), (2, 3), (3, 2)]