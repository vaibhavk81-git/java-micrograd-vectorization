package ai.micrograd.tensor;

/**
 * TensorOps provides static factory methods for tensor operations with automatic differentiation.
 * 
 * <p>Each operation:
 * <ul>
 *   <li>Performs the forward computation</li>
 *   <li>Wires the computation graph (parents, op name)</li>
 *   <li>Defines the backward function for gradient computation</li>
 * </ul>
 * 
 * <p><b>Gradient Accumulation:</b> All backward functions use += to accumulate gradients,
 * supporting the multivariate chain rule.
 * 
 * @author Vaibhav Khare
 */
public final class TensorOps {
    
    private TensorOps() {} // Prevent instantiation
    
    // ========== Element-wise Operations ==========
    
    /**
     * Element-wise addition: out = a + b
     * 
     * <p>Shapes must match exactly.
     * 
     * <p><b>Gradient:</b> dL/da = dL/dout, dL/db = dL/dout
     * 
     * @param a first tensor
     * @param b second tensor
     * @return result tensor
     * @throws IllegalArgumentException if shapes don't match
     */
    public static Tensor add(Tensor a, Tensor b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch for add: (%d, %d) vs (%d, %d)",
                    a.rows(), a.cols(), b.rows(), b.cols()));
        }
        
        boolean requiresGrad = a.requiresGrad() || b.requiresGrad();
        Tensor out = new Tensor(a.rows(), a.cols(), requiresGrad);
        
        // Forward: out = a + b
        for (int i = 0; i < out.data().length; i++) {
            out.data()[i] = a.data()[i] + b.data()[i];
        }
        
        // Backward: da += dout, db += dout
        if (requiresGrad) {
            out.setParents(a, b);
            out.setOp("add");
            out.setBackwardFn(() -> {
                if (a.requiresGrad()) {
                    for (int i = 0; i < out.data().length; i++) {
                        a.grad()[i] += out.grad()[i];
                    }
                }
                if (b.requiresGrad()) {
                    for (int i = 0; i < out.data().length; i++) {
                        b.grad()[i] += out.grad()[i];
                    }
                }
            });
        }
        
        return out;
    }
    
    /**
     * Element-wise multiplication: out = a * b
     * 
     * <p>Shapes must match exactly.
     * 
     * <p><b>Gradient:</b> dL/da = b * dL/dout, dL/db = a * dL/dout
     * 
     * @param a first tensor
     * @param b second tensor
     * @return result tensor
     * @throws IllegalArgumentException if shapes don't match
     */
    public static Tensor mul(Tensor a, Tensor b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch for mul: (%d, %d) vs (%d, %d)",
                    a.rows(), a.cols(), b.rows(), b.cols()));
        }
        
        boolean requiresGrad = a.requiresGrad() || b.requiresGrad();
        Tensor out = new Tensor(a.rows(), a.cols(), requiresGrad);
        
        // Forward: out = a * b
        for (int i = 0; i < out.data().length; i++) {
            out.data()[i] = a.data()[i] * b.data()[i];
        }
        
        // Backward: da += b * dout, db += a * dout
        if (requiresGrad) {
            out.setParents(a, b);
            out.setOp("mul");
            out.setBackwardFn(() -> {
                if (a.requiresGrad()) {
                    for (int i = 0; i < out.data().length; i++) {
                        a.grad()[i] += b.data()[i] * out.grad()[i];
                    }
                }
                if (b.requiresGrad()) {
                    for (int i = 0; i < out.data().length; i++) {
                        b.grad()[i] += a.data()[i] * out.grad()[i];
                    }
                }
            });
        }
        
        return out;
    }
    
    /**
     * Hyperbolic tangent activation: out = tanh(x)
     * 
     * <p><b>Gradient:</b> dL/dx = (1 - tanh²(x)) * dL/dout = (1 - out²) * dL/dout
     * 
     * @param x input tensor
     * @return result tensor
     */
    public static Tensor tanh(Tensor x) {
        Tensor out = new Tensor(x.rows(), x.cols(), x.requiresGrad());
        
        // Forward: out = tanh(x)
        for (int i = 0; i < out.data().length; i++) {
            out.data()[i] = Math.tanh(x.data()[i]);
        }
        
        // Backward: dx += (1 - out²) * dout
        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("tanh");
            out.setBackwardFn(() -> {
                for (int i = 0; i < out.data().length; i++) {
                    double t = out.data()[i];
                    x.grad()[i] += (1.0 - t * t) * out.grad()[i];
                }
            });
        }
        
        return out;
    }
    
    /**
     * ReLU activation: out = max(0, x)
     * 
     * <p><b>Gradient:</b> dL/dx = (x > 0 ? 1 : 0) * dL/dout
     * 
     * @param x input tensor
     * @return result tensor
     */
    public static Tensor relu(Tensor x) {
        Tensor out = new Tensor(x.rows(), x.cols(), x.requiresGrad());
        
        // Forward: out = max(0, x)
        for (int i = 0; i < out.data().length; i++) {
            out.data()[i] = Math.max(0.0, x.data()[i]);
        }
        
        // Backward: dx += (x > 0 ? 1 : 0) * dout
        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("relu");
            out.setBackwardFn(() -> {
                for (int i = 0; i < out.data().length; i++) {
                    if (x.data()[i] > 0) {
                        x.grad()[i] += out.grad()[i];
                    }
                }
            });
        }
        
        return out;
    }
    
    // ========== Broadcast Operations ==========
    
    /**
     * Broadcast row-vector addition: out = matrix + rowVec
     * 
     * <p>Adds a row vector (1×n) to each row of a matrix (m×n).
     * 
     * <p><b>Shapes:</b> matrix: (m×n), rowVec: (1×n), out: (m×n)
     * 
     * <p><b>Gradient:</b>
     * <ul>
     *   <li>dL/dmatrix = dL/dout (same shape)</li>
     *   <li>dL/drowVec = sum over rows of dL/dout (reduce to 1×n)</li>
     * </ul>
     * 
     * @param matrix matrix tensor (m×n)
     * @param rowVec row vector tensor (1×n)
     * @return result tensor (m×n)
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public static Tensor addRowVector(Tensor matrix, Tensor rowVec) {
        if (rowVec.rows() != 1) {
            throw new IllegalArgumentException(
                String.format("rowVec must have 1 row, got: %d", rowVec.rows()));
        }
        if (matrix.cols() != rowVec.cols()) {
            throw new IllegalArgumentException(
                String.format("Column mismatch: matrix has %d cols, rowVec has %d cols",
                    matrix.cols(), rowVec.cols()));
        }
        
        boolean requiresGrad = matrix.requiresGrad() || rowVec.requiresGrad();
        Tensor out = new Tensor(matrix.rows(), matrix.cols(), requiresGrad);
        
        // Forward: out[r, c] = matrix[r, c] + rowVec[0, c]
        for (int r = 0; r < out.rows(); r++) {
            for (int c = 0; c < out.cols(); c++) {
                int idx = r * out.cols() + c;
                out.data()[idx] = matrix.data()[idx] + rowVec.data()[c];
            }
        }
        
        // Backward: dmatrix += dout, drowVec += sum_rows(dout)
        if (requiresGrad) {
            out.setParents(matrix, rowVec);
            out.setOp("addRowVector");
            out.setBackwardFn(() -> {
                if (matrix.requiresGrad()) {
                    for (int i = 0; i < out.data().length; i++) {
                        matrix.grad()[i] += out.grad()[i];
                    }
                }
                if (rowVec.requiresGrad()) {
                    // Sum over rows
                    for (int r = 0; r < out.rows(); r++) {
                        for (int c = 0; c < out.cols(); c++) {
                            int idx = r * out.cols() + c;
                            rowVec.grad()[c] += out.grad()[idx];
                        }
                    }
                }
            });
        }
        
        return out;
    }
    
    // ========== Reduction Operations ==========
    
    /**
     * Sum along an axis.
     * 
     * <p><b>axis=0:</b> sum over rows, output shape (1×n)
     * <p><b>axis=1:</b> sum over columns, output shape (m×1)
     * 
     * <p><b>Gradient:</b> Broadcast upstream gradient back to input shape
     * 
     * @param x input tensor (m×n)
     * @param axis axis to sum along (0 or 1)
     * @return result tensor
     * @throws IllegalArgumentException if axis is not 0 or 1
     */
    public static Tensor sum(Tensor x, int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got: " + axis);
        }
        
        Tensor out;
        if (axis == 0) {
            // Sum over rows -> (1×n)
            out = new Tensor(1, x.cols(), x.requiresGrad());
            for (int c = 0; c < x.cols(); c++) {
                double sum = 0.0;
                for (int r = 0; r < x.rows(); r++) {
                    sum += x.data()[r * x.cols() + c];
                }
                out.data()[c] = sum;
            }
        } else {
            // Sum over columns -> (m×1)
            out = new Tensor(x.rows(), 1, x.requiresGrad());
            for (int r = 0; r < x.rows(); r++) {
                double sum = 0.0;
                for (int c = 0; c < x.cols(); c++) {
                    sum += x.data()[r * x.cols() + c];
                }
                out.data()[r] = sum;
            }
        }
        
        // Backward: broadcast dout back to input shape
        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("sum(axis=" + axis + ")");
            out.setBackwardFn(() -> {
                if (axis == 0) {
                    // Broadcast (1×n) -> (m×n)
                    for (int r = 0; r < x.rows(); r++) {
                        for (int c = 0; c < x.cols(); c++) {
                            x.grad()[r * x.cols() + c] += out.grad()[c];
                        }
                    }
                } else {
                    // Broadcast (m×1) -> (m×n)
                    for (int r = 0; r < x.rows(); r++) {
                        for (int c = 0; c < x.cols(); c++) {
                            x.grad()[r * x.cols() + c] += out.grad()[r];
                        }
                    }
                }
            });
        }
        
        return out;
    }
    
    /**
     * Mean along an axis.
     * 
     * <p><b>axis=0:</b> mean over rows, output shape (1×n)
     * <p><b>axis=1:</b> mean over columns, output shape (m×1)
     * 
     * <p><b>Gradient:</b> Broadcast upstream gradient back to input shape, scaled by 1/count
     * 
     * @param x input tensor (m×n)
     * @param axis axis to average along (0 or 1)
     * @return result tensor
     * @throws IllegalArgumentException if axis is not 0 or 1
     */
    public static Tensor mean(Tensor x, int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got: " + axis);
        }
        
        Tensor sum = sum(x, axis);
        int count = (axis == 0) ? x.rows() : x.cols();
        
        Tensor out = new Tensor(sum.rows(), sum.cols(), x.requiresGrad());
        for (int i = 0; i < out.data().length; i++) {
            out.data()[i] = sum.data()[i] / count;
        }
        
        // Backward: broadcast dout back to input shape, scaled by 1/count
        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("mean(axis=" + axis + ")");
            double scale = 1.0 / count;
            out.setBackwardFn(() -> {
                if (axis == 0) {
                    // Broadcast (1×n) -> (m×n)
                    for (int r = 0; r < x.rows(); r++) {
                        for (int c = 0; c < x.cols(); c++) {
                            x.grad()[r * x.cols() + c] += out.grad()[c] * scale;
                        }
                    }
                } else {
                    // Broadcast (m×1) -> (m×n)
                    for (int r = 0; r < x.rows(); r++) {
                        for (int c = 0; c < x.cols(); c++) {
                            x.grad()[r * x.cols() + c] += out.grad()[r] * scale;
                        }
                    }
                }
            });
        }
        
        return out;
    }
    
    // ========== Matrix Multiplication ==========
    
    /**
     * Matrix multiplication: out = a @ b
     * 
     * <p><b>Shapes:</b> a: (m×k), b: (k×n), out: (m×n)
     * 
     * <p><b>Gradient:</b>
     * <ul>
     *   <li>dL/da = dL/dout @ b^T</li>
     *   <li>dL/db = a^T @ dL/dout</li>
     * </ul>
     * 
     * @param a left matrix (m×k)
     * @param b right matrix (k×n)
     * @return result matrix (m×n)
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.cols() != b.rows()) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch for matmul: (%d, %d) @ (%d, %d)",
                    a.rows(), a.cols(), b.rows(), b.cols()));
        }
        
        int m = a.rows();
        int k = a.cols();
        int n = b.cols();
        
        boolean requiresGrad = a.requiresGrad() || b.requiresGrad();
        Tensor out = new Tensor(m, n, requiresGrad);
        
        // Forward: out[i,j] = sum_k a[i,k] * b[k,j]
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    sum += a.data()[i * k + kk] * b.data()[kk * n + j];
                }
                out.data()[i * n + j] = sum;
            }
        }
        
        // Backward: da += dout @ b^T, db += a^T @ dout
        if (requiresGrad) {
            out.setParents(a, b);
            out.setOp("matmul");
            out.setBackwardFn(() -> {
                if (a.requiresGrad()) {
                    // da += dout @ b^T
                    // da[i,k] += sum_j dout[i,j] * b[k,j]
                    for (int i = 0; i < m; i++) {
                        for (int kk = 0; kk < k; kk++) {
                            double sum = 0.0;
                            for (int j = 0; j < n; j++) {
                                sum += out.grad()[i * n + j] * b.data()[kk * n + j];
                            }
                            a.grad()[i * k + kk] += sum;
                        }
                    }
                }
                if (b.requiresGrad()) {
                    // db += a^T @ dout
                    // db[k,j] += sum_i a[i,k] * dout[i,j]
                    for (int kk = 0; kk < k; kk++) {
                        for (int j = 0; j < n; j++) {
                            double sum = 0.0;
                            for (int i = 0; i < m; i++) {
                                sum += a.data()[i * k + kk] * out.grad()[i * n + j];
                            }
                            b.grad()[kk * n + j] += sum;
                        }
                    }
                }
            });
        }
        
        return out;
    }
}

