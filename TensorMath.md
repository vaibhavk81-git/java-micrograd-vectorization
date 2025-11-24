# Tensor Math Reference

Detailed math notes for every tensor operation implemented in the Java makemore backend. Each section explains the forward computation, an illustrative numeric example, and how gradients propagate (matching PyTorch semantics).

## Add (`TensorOps.add`)
- **Forward:** `out[i,j] = a[i,j] + b[i,j]`
- **Example:**  
  `a = [[1,2],[3,4]]`, `b = [[5,6],[7,8]]` → `out = [[6,8],[10,12]]`
- **Backward:** `dL/da = dL/dout`, `dL/db = dL/dout`

## Elementwise Multiply (`TensorOps.mul`)
- **Forward:** `out[i,j] = a[i,j] * b[i,j]`
- **Example:**  
  `a = [[1,2],[3,4]]`, `b = [[2,0],[1,5]]` → `out = [[2,0],[3,20]]`
- **Backward:** `dL/da = (dL/dout) ⊙ b`, `dL/db = (dL/dout) ⊙ a`

## Tanh (`TensorOps.tanh`)
- **Forward:** `out[i,j] = tanh(x[i,j])`
- **Example:** `x = [0, 1]` → `out ≈ [0, 0.7616]`
- **Backward:** `dL/dx = (1 - tanh(x)^2) ⊙ dL/dout`

## ReLU (`TensorOps.relu`)
- **Forward:** `out[i,j] = max(0, x[i,j])`
- **Example:** `x = [-1, 0.5]` → `out = [0, 0.5]`
- **Backward:** Pass gradient only where `x > 0`

## Add Row Vector (`TensorOps.addRowVector`)
- **Forward:** Broadcast a row vector across rows: `out[r,c] = matrix[r,c] + row[c]`
- **Example:**  
  `matrix = [[1,2],[3,4]]`, `row = [10,20]` → `out = [[11,22],[13,24]]`
- **Backward:** Matrix receives full gradient, row accumulates column-wise sums of `dL/dout`

## Sum (`TensorOps.sum`)
- **Forward:** Reduce along axis:  
  `axis=0` (rows) → `out[0,c] = Σ_r x[r,c]`  
  `axis=1` (cols) → `out[r,0] = Σ_c x[r,c]`
- **Example:** `x = [[1,2],[3,4]]`  
  `axis=0` → `[4,6]`; `axis=1` → `[3,7]`
- **Backward:** Broadcast reduced gradient back to original shape

## Mean (`TensorOps.mean`)
- **Forward:** `mean = sum(axis)/count`
- **Example:** `x = [[1,2],[3,4]]`, `axis=0` → `[2,3]`
- **Backward:** Same as sum but scaled by `1/count`

## MatMul (`TensorOps.matmul`)
- **Forward:** Standard matrix multiplication `(m×k) @ (k×n) = (m×n)`
- **Example:**  
  `a = [[1,2],[3,4]]`, `b = [[5,6],[7,8]]` → `[[19,22],[43,50]]`
- **Backward:**  
  `dL/da = dL/dout @ bᵀ`  
  `dL/db = aᵀ @ dL/dout`

## Embedding (`TensorOps.embedding`)
- **Forward (mirrors `CpuBackend.embedding` step-by-step):**
  1. Flatten `indices` in row-major order (`blockCount = batchSize * seqLen`).
  2. For each flattened position `i`, read the integer token `idx`.
  3. Copy the slice `weight[idx, :]` into `out[i, :]`.
  4. Store `idx` in `gatheredIdx[i]` for backward.
  Output shape: `(batchSize * seqLen, embedDim)`.
- **Example:**  
  `weight = [[1,0],[0,1],[2,2]]`, `indices = [[0,2],[1,0]]`  
  Flattened indices `[0,2,1,0]` → output rows `[[1,0],[2,2],[0,1],[1,0]]`.
- **Backward:** For each output grad row `gradOut[i]`, look up its cached token `idx = gatheredIdx[i]` and accumulate (`+=`) `gradOut[i]` into `weightGrad[idx]`. Repeated tokens therefore add up automatically (identical to PyTorch’s scatter-add).
- **PyTorch parity:** `torch.nn.functional.embedding`.

## Cross Entropy (`TensorOps.crossEntropy`)
- **Forward (exactly as in `CpuBackend.crossEntropy`):**
  1. For each batch row `i`, compute `maxLogit = max_j logits[i,j]`.
  2. Compute `exp_j = exp(logits[i,j] - maxLogit)`, sum them to `sumExp`.
  3. Normalize `prob[i,j] = exp_j / sumExp` and cache it for backward.
  4. Loss for row `i`: `ℓ_i = -log(prob[i, target_i]) = log(sumExp) + maxLogit - logits[i,target_i]`.
  5. Return the mean `(1/batch) Σ_i ℓ_i`.
- **Example:** `logits = [[2,0]], target=0`  
  `max=2`, `exp = [1, e^{-2}]`, `sumExp ≈ 1.1353`, `prob ≈ [0.881, 0.119]`, `loss ≈ 0.1278`.
- **Backward:** With cached `prob`, set `grad[i,j] = prob[i,j] - 1` when `j` is the target, otherwise `prob[i,j]`. Scale by `1/batch` and multiply by upstream loss grad (`outGrad[0]`). This matches `softmax(logits) - one_hot(target)`.
- **PyTorch parity:** `torch.nn.functional.cross_entropy`.

## Reshape (`TensorOps.reshape`)
- **Forward:** Checks that `newRows * newCols == oldRows * oldCols`, then copies the linear buffer into a tensor wrapper with the new shape. Conceptually equivalent to PyTorch’s `view`. No data rearrangement beyond reinterpretation.
- **Example:** `(2×3)` matrix  
  ```
  [[1,2,3],
   [4,5,6]]
  ```  
  reshaped to `(3×2)` becomes  
  ```
  [[1,2],
   [3,4],
   [5,6]]
  ```
- **Backward:** During `backward`, the grad buffer from the reshape output is copied back into the parent’s grad array (same linear order), effectively an identity mapping.
- **PyTorch parity:** `tensor.view`.

## SGD Step (`TensorBackend.sgdStep`)
- **Algorithm in `CpuBackend.sgdStep`:**
  1. Access the parameter’s data buffer (`double[]` or `float[]`).
  2. Access the gradient buffer of the same size.
  3. For every element `i`, compute `data[i] -= lr * grad[i]`.
  4. No momentum/weight decay is applied—this is the vanilla PyTorch SGD update.
- **Example:** `param=[1.0, 0.5]`, `grad=[0.2, -0.4]`, `lr=0.1`  
  → `[0.98, 0.54]`.
- **Note:** Gradients remain stored after the update; the optimizer caller should zero them (e.g., `SGD.zeroGrad(...)`) before the next backward pass.

