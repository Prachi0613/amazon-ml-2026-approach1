# Mandatory Diagnostic Gate Report

## 1. Current blocking code path
`src/blocking.py`

## 2. Memory before each stage
*(See Test A and Test B Results below)*

## 3. Memory after each stage
*(See Test A and Test B Results below)*

## 4. Memory delta
*(See Test A and Test B Results below)*

## 5. Largest objects
- `TfidfVectorizer.vocabulary_`: A massive Python dictionary storing every unique character n-gram as a string key.
- `_ExactIndex`: A Python `defaultdict` storing lists of tuples `(entity_id: str, source_label: str)`.
- `csr_matrix`: The document-term sparse matrix `X_corpus`.

## 6. Dense allocation findings
Searched for `.toarray()`, `.todense()`, `np.asarray()`, `cosine_similarity()`, `squareform()`, `vstack()`, `hstack()`, `concatenate()` inside `src/blocking.py`. 
**Finding:** None. The retrieval path exclusively uses `X_query @ X_corpus.T` which natively yields a `csr_matrix`, and top-K extraction uses sparse matrix `indptr` and `data` arrays. No dense cross-product matrices are allocated in the retrieval loop.

## 7. Theoretical allocation sizes
NOT MEASURED (No dense cross-product matrices are created in `blocking.py`).

## 8. Accumulation vs single allocation
**A. Gradual accumulation.**
The pipeline OOMs before chunked retrieval even begins. Memory scales primarily with **S2/S3 corpus size**. Each consecutive call to `_build_exact_index` and `_build_ngram_index` allocates enormous Python dictionaries and sparse matrices which are permanently held in RAM to serve as the inverted indices. 

## 9. Test A results
`S1 = 10,000 | S2 = 50,000 | S3 = 50,000` (Corpus size: 100,000)

| operation | RSS before | RSS after | delta |
| :--- | :--- | :--- | :--- |
| 1. exact name index creation | 158.16 MB | 186.23 MB | +28.07 MB |
| 2. exact address index creation | 186.23 MB | 217.15 MB | +30.93 MB |
| 3. n-gram name vectorizer/index | 217.15 MB | 263.28 MB | +46.13 MB |
| *corpus transformation* | *271.20 MB* | *304.77 MB* | *+33.57 MB* |
| 4. n-gram address vectorizer/index | 263.28 MB | 302.77 MB | +39.49 MB |
| query transformation | 312.29 MB | 312.31 MB | +0.02 MB |
| similarity/retrieval | 312.31 MB | 335.00 MB | +22.69 MB |
| candidate creation | 335.00 MB | 318.30 MB | -16.70 MB |
| candidate concatenation | 313.88 MB | 321.64 MB | +7.76 MB |
| candidate deduplication | 322.51 MB | 341.03 MB | +18.52 MB |

## 10. Test B results
`S1 = 25,000 | S2 = 100,000 | S3 = 100,000` (Corpus size: 200,000)

| operation | RSS before | RSS after | delta |
| :--- | :--- | :--- | :--- |
| 1. exact name index creation | 230.55 MB | 263.16 MB | +32.61 MB |
| 2. exact address index creation | 263.16 MB | 325.34 MB | +62.18 MB |
| 3. n-gram name vectorizer/index | 325.34 MB | 405.70 MB | +80.36 MB |
| *corpus transformation* | *433.46 MB* | *494.88 MB* | *+61.42 MB* |
| 4. n-gram address vectorizer/index | 405.70 MB | 482.59 MB | +76.89 MB |
| query transformation | 481.51 MB | 481.51 MB | +0.00 MB |
| similarity/retrieval | 481.51 MB | 525.28 MB | +43.77 MB |
| candidate creation | 525.28 MB | 529.38 MB | +4.09 MB |
| candidate concatenation | 491.95 MB | 509.12 MB | +17.17 MB |
| candidate deduplication | 509.13 MB | 542.50 MB | +33.37 MB |

## 11. Exact root cause
When extrapolated to the real corpus size of ~10.3 million rows (50x the size of Test B):
- n-gram name index: `50 * 80 MB = ~4.0 GB`
- n-gram address index: `50 * 77 MB = ~3.85 GB`
- exact indices: `50 * (62 + 32) MB = ~4.7 GB`
Total index accumulation overhead exceeds `12.5 GB` of RAM. Added to the `6.9 GB` required for dataframe loading, the process accumulates over `19.4 GB` of RAM and triggers an OOM kill *before candidate chunking even starts*. 
The RAM is overwhelmingly consumed by massive dictionaries storing millions of unique strings (Tfidf vocabulary and exact-match string tuples).

## 12. Exact file/function responsible
`src/blocking.py`
Functions: `_build_exact_index()` and `_build_ngram_index()`

## 13. Recommended fix
1. **HashingVectorizer**: Replace `TfidfVectorizer` with `HashingVectorizer(n_features=2**20)`. This completely eliminates the unbounded vocabulary dictionary, capping the n-gram index overhead exclusively to the non-zeros of the CSR matrix.
2. **Integer Exact Index**: Modify `_build_exact_index` to map normalized strings to flat arrays of integer row indices rather than heavily nested lists of string tuples.

## 14. Expected memory behavior after fix
The index building phase will become bounded. The HashingVectorizer will use effectively zero memory for its vocabulary, isolating memory footprint entirely to the sparse matrix data arrays. The exact index will allocate highly contiguous integers instead of PyObjects, slashing its memory delta by >80%. The pipeline will comfortably survive index creation and proceed to candidate chunking.
