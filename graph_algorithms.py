"""
Part 1: Huffman coding (encode_text / decode_text)
Part 2: tree isomorphism (are_isomorphic / mapping_isomorphic)

Implements:
  - encode_text(text: str) -> tuple[str, dict[str, str]]
  - decode_text(encoded_text: str, codes: dict[str, str]) -> str

Features and requirements satisfied:
  • Huffman coding (optimal prefix-free code).
  • Empty string ⇒ ("", {}).
  • Single unique character ⇒ code '0'; encoded text is '0' * len(text).
  • Return the code dictionary sorted by:
      - descending frequency in the given text,
      - then lexicographic order (for ties).
  • Decoding runs in O(m) for m bits.
"""

from __future__ import annotations
from collections import Counter, OrderedDict, deque
from typing import Dict, Tuple, Optional, List
import heapq


class _Node:
    """
    Internal node for a Huffman tree.
    Leaves have .char set, internal nodes have .char is None.

    Attributes:
        char: Optional[str] — character at this leaf; None for internal nodes
        freq: int           — frequency (or sum of children frequencies)
        left, right: _Node  — children (None for leaves)
        _tie: int           — stable tie-breaker for heap ordering
    """
    __slots__ = ("char", "freq", "left", "right", "_tie")

    def __init__(self, char: Optional[str], freq: int, tie: int = 0,
                 left: Optional["_Node"] = None, right: Optional["_Node"] = None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        self._tie = tie

    def __lt__(self, other: "_Node") -> bool:
        # heapq needs a strict weak ordering
        if self.freq != other.freq:
            return self.freq < other.freq
        return self._tie < other._tie


def _build_huffman_tree(text: str) -> Optional[_Node]:
    """
    Builds a Huffman tree from the given text.
    Returns the root node, or None if the text is empty.
    """
    freq = Counter(text)
    if not freq:
        return None

    # Special case: only one unique character.
    if len(freq) == 1:
        ch, f = next(iter(freq.items()))
        root = _Node(None, f)
        root.left = _Node(ch, f)   # ensure code '0' is assigned to the single char
        return root

    heap: List[_Node] = []
    tie = 0
    for ch, f in freq.items():
        heap.append(_Node(ch, f, tie))
        tie += 1
    heapq.heapify(heap)

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = _Node(None, a.freq + b.freq, tie, a, b)
        tie += 1
        heapq.heappush(heap, parent)

    return heap[0]


def _build_codes(root: Optional[_Node]) -> Dict[str, str]:
    """
    Traverses the Huffman tree to build a mapping char -> code.
    For the single-character case, assigns code '0'.
    """
    codes: Dict[str, str] = {}
    if root is None:
        return codes

    def dfs(node: Optional[_Node], prefix: str) -> None:
        if node is None:
            return
        if node.char is not None:  # leaf
            codes[node.char] = prefix if prefix != "" else "0"
            return
        dfs(node.left, prefix + "0")
        dfs(node.right, prefix + "1")

    dfs(root, "")
    return codes


def encode_text(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Problem 1A: Encode a text using optimal (Huffman) coding.

    Args:
        text: Input string over an arbitrary alphabet.

    Returns:
        (encoded_text, codes):
            encoded_text: the binary string (as '0'/'1' characters).
            codes: dict mapping each character to its binary code,
                   sorted by descending frequency in the given text,
                   then lexicographically for ties.

    Edge cases:
        • text == ""  -> ("", {})
        • single unique character 'x'  -> codes = {'x': '0'}, encoded_text = '0' * len(text)
    """
    if text == "":
        return "", {}

    root = _build_huffman_tree(text)
    codes_unsorted = _build_codes(root)

    # Encode
    encoded_text = "".join(codes_unsorted[ch] for ch in text)

    # Sort the code dictionary by (−frequency, lexicographic char)
    freq = Counter(text)
    items_sorted = sorted(codes_unsorted.items(), key=lambda kv: (-freq[kv[0]], kv[0]))
    codes_sorted = OrderedDict(items_sorted)  # preserves order in Python 3.7+

    return encoded_text, dict(codes_sorted)


def decode_text(encoded_text: str, codes: Dict[str, str]) -> str:
    """
    Problem 1B: Decode a binary string using prefix-free codes.

    Args:
        encoded_text: binary string made of '0'/'1'.
        codes: mapping from character to its binary code.

    Returns:
        The original decoded text.

    Raises:
        ValueError: if `encoded_text` is inconsistent with the provided `codes`.

    Complexity:
        • Build tree: O(k) for k codes.
        • Decode scan: O(m) for m bits in encoded_text.
    """
    if not encoded_text and not codes:
        return ""

    # Build a decoding trie
    TRIE_END = "__END__"
    trie = {}
    for ch, code in codes.items():
        node = trie
        for bit in code:
            if bit not in ("0", "1"):
                raise ValueError("Codes must contain only '0' and '1'.")
            node = node.setdefault(bit, {})
        if TRIE_END in node:
            raise ValueError("Duplicate code in codes dictionary.")
        node[TRIE_END] = ch

    # Decode by traversing the trie
    result_chars: List[str] = []
    node = trie
    for bit in encoded_text:
        if bit not in ("0", "1"):
            raise ValueError("Encoded text must contain only '0' and '1'.")
        node = node.get(bit)
        if node is None:
            raise ValueError("Encoded text is invalid for the given codes.")
        ch = node.get(TRIE_END)
        if ch is not None:
            result_chars.append(ch)
            node = trie  # reset for next character

    if node is not trie:
        # ended mid-code
        raise ValueError("Encoded text ended in the middle of a codeword.")

    return "".join(result_chars)


"""
Problems 2A, 2B  —  tree isomorphism (unrooted trees)

Algorithm (AHU with canonical integer ids, fully iterative):
  1. Convert the adjacency matrix to adjacency lists.           O(n^2) (reading the input)
  2. Find the center(s) of each tree by peeling leaves.          O(n)
  3. Root each tree at a center; process vertices bottom-up. The id of a
     vertex = id of the sorted tuple of its children's ids (a dict shared
     by both trees gives equal ids to isomorphic rooted subtrees).  O(n log n)
  4. Trees are isomorphic  <=>  tree_1 rooted at its center has the same id as
     tree_2 rooted at one of ITS centers (a tree has at most 2 centers).
  5. Mapping: walk both trees top-down together; match children with equal ids.   O(n)

Everything is iterative (no recursion), so deep trees (e.g. a path) are safe.
"""


def adjacency_matrix_to_list(matrix):
    n = len(matrix)
    adj = [[] for _ in range(n)]
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            if row[j]:
                adj[i].append(j)
    return adj


def find_tree_centers(adj):
    n = len(adj)
    if n == 0:
        return []
    degree = [len(nb) for nb in adj]
    leaves = deque(i for i in range(n) if degree[i] <= 1)
    removed = len(leaves)
    while removed < n:
        new_leaves = deque()
        for leaf in leaves:
            for nb in adj[leaf]:
                degree[nb] -= 1
                if degree[nb] == 1:
                    new_leaves.append(nb)
        removed += len(new_leaves)
        leaves = new_leaves
    return list(leaves)


def _canonical_ids(adj, root, table):
    """
    Iteratively computes a canonical integer id for every vertex of the tree
    rooted at `root`. Returns (ids, parent, order). `table` maps a sorted tuple
    of children ids -> id and is shared between trees so ids are comparable.
    """
    n = len(adj)
    parent = [-1] * n
    order = [root]
    for u in order:                      # BFS (list grows while iterating)
        for w in adj[u]:
            if w != parent[u]:
                parent[w] = u
                order.append(w)
    kids = [[] for _ in range(n)]
    ids = [0] * n
    for u in reversed(order):            # children before parents
        kids[u].sort()
        key = tuple(kids[u])
        i = table.get(key)
        if i is None:
            i = len(table)
            table[key] = i
        ids[u] = i
        if parent[u] != -1:
            kids[parent[u]].append(i)
    return ids, parent, order


def are_isomorphic(tree_1: List[List[bool]], tree_2: List[List[bool]]) -> bool:
    n = len(tree_1)
    if n != len(tree_2):
        return False
    if n == 0:
        return True
    adj1 = adjacency_matrix_to_list(tree_1)
    adj2 = adjacency_matrix_to_list(tree_2)
    if sum(map(len, adj1)) != sum(map(len, adj2)):
        return False
    c1 = find_tree_centers(adj1)
    c2 = find_tree_centers(adj2)
    if len(c1) != len(c2):
        return False
    table = {}
    ids1, _, _ = _canonical_ids(adj1, c1[0], table)
    target = ids1[c1[0]]
    for c in c2:
        ids2, _, _ = _canonical_ids(adj2, c, table)
        if ids2[c] == target:
            return True
    return False


def mapping_isomorphic(tree_1: List[List[bool]], tree_2: List[List[bool]]) -> List[Tuple[int, int]]:
    n = len(tree_1)
    if n != len(tree_2) or n == 0:
        return []
    adj1 = adjacency_matrix_to_list(tree_1)
    adj2 = adjacency_matrix_to_list(tree_2)
    if sum(map(len, adj1)) != sum(map(len, adj2)):
        return []
    c1 = find_tree_centers(adj1)
    c2 = find_tree_centers(adj2)
    if len(c1) != len(c2):
        return []

    table = {}
    ids1, par1, _ = _canonical_ids(adj1, c1[0], table)
    root1 = c1[0]
    root2 = None
    for c in c2:
        ids2, par2, _ = _canonical_ids(adj2, c, table)
        if ids2[c] == ids1[root1]:
            root2 = c
            break
    if root2 is None:
        return []

    mapping = []
    stack = [(root1, root2)]
    while stack:
        u, v = stack.pop()
        mapping.append((u, v))
        ch1 = sorted((w for w in adj1[u] if w != par1[u]), key=ids1.__getitem__)
        ch2 = sorted((w for w in adj2[v] if w != par2[v]), key=ids2.__getitem__)
        stack.extend(zip(ch1, ch2))      # equal ids pair up, since the multisets are equal
    mapping.sort()
    return mapping

if __name__ == "__main__":
    # ===== Part 1 examples =====
    for label, t in (("Empty string", ""), ("Single character", "aaaaaa"),
                     ("Longer text", "this is an example of a huffman tree")):
        enc, codes = encode_text(t)
        dec = decode_text(enc, codes)
        assert dec == t
        print(f"[{label}] text: {t!r}")
        print("encoded:", repr(enc))
        print("codes:  ", codes)
        print("decoded:", repr(dec))
        print()

    # ===== Part 2 examples =====
    # Empty trees
    print(are_isomorphic([], []))                        # True
    print(mapping_isomorphic([], []))                    # []

    # Single node
    print(are_isomorphic([[False]], [[False]]))          # True
    print(mapping_isomorphic([[False]], [[False]]))      # [(0, 0)]

    # Isomorphic 4-vertex trees
    T1 = [[False, True, True, False],
          [True, False, False, False],
          [True, False, False, True],
          [False, False, True, False]]
    T2 = [[False, True, False, True],
          [True, False, False, False],
          [False, False, False, True],
          [True, False, True, False]]
    print(are_isomorphic(T1, T2))                        # True
    print(mapping_isomorphic(T1, T2))                    # [(0, 0), (1, 1), (2, 3), (3, 2)]

    # Non-isomorphic 7-vertex trees
    tree_1 = [
        [False, True,  True,  False, False, True,  False],
        [True,  False, False, True,  True,  False, False],
        [True,  False, False, False, False, False, False],
        [False, True,  False, False, False, False, False],
        [False, True,  False, False, False, False, False],
        [True,  False, False, False, False, False, True],
        [False, False, False, False, False, True,  False]]
    tree_2 = [
        [False, True,  True,  False, False, False, False],
        [True,  False, False, True,  True,  False, False],
        [True,  False, False, False, False, True,  False],
        [False, True,  False, False, False, False, False],
        [False, True,  False, False, False, False, True],
        [False, False, True,  False, False, False, False],
        [False, False, False, False, True,  False, False]]
    print(are_isomorphic(tree_1, tree_2))                # False
    print(mapping_isomorphic(tree_1, tree_2))            # []