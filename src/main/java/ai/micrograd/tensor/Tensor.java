package ai.micrograd.tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;

/**
 * Tensor represents a 2-D dense tensor with automatic differentiation capabilities.
 * 
 * <p>This class uses flat double[] storage for efficiency and supports:
 * <ul>
 *   <li>Element-wise operations (add, mul, tanh, relu)</li>
 *   <li>Broadcast operations (addRowVector)</li>
 *   <li>Matrix multiplication (matmul)</li>
 *   <li>Reductions (sum, mean along axis)</li>
 *   <li>Automatic gradient computation via backpropagation</li>
 * </ul>
 * 
 * <p><b>Storage:</b> Data is stored in row-major order in a flat double[] array.
 * For a tensor with shape (m, n), element at row r, col c is at index r*n + c.
 * 
 * <p><b>Gradients:</b> Gradients accumulate (+=) during backpropagation to support
 * the multivariate chain rule.
 * 
 * @author Vaibhav Khare
 */
public final class Tensor {
    
    private final int rows;
    private final int cols;
    private final double[] data;
    private final double[] grad;
    private final boolean requiresGrad;
    private final Precision precision;
    
    // Autograd graph
    private Tensor[] parents;
    private Runnable backwardFn;
    private String op;
    private String label;
    
    /**
     * Creates a tensor with the specified shape and precision.
     * 
     * @param rows number of rows
     * @param cols number of columns
     * @param precision precision mode (FP64 or FP32)
     * @param requiresGrad whether to track gradients
     * @throws IllegalArgumentException if rows or cols < 1
     * @throws UnsupportedOperationException if precision is FP32 (not yet implemented)
     */
    public Tensor(int rows, int cols, Precision precision, boolean requiresGrad) {
        if (rows < 1 || cols < 1) {
            throw new IllegalArgumentException(
                String.format("Tensor dimensions must be positive, got: (%d, %d)", rows, cols));
        }
        
        if (precision == Precision.FP32) {
            throw new UnsupportedOperationException(
                "FP32 precision is not yet implemented. Use Precision.FP64.");
        }
        
        this.rows = rows;
        this.cols = cols;
        this.precision = precision;
        this.requiresGrad = requiresGrad;
        this.data = new double[rows * cols];
        this.grad = requiresGrad ? new double[rows * cols] : null;
        this.parents = new Tensor[0];
        this.backwardFn = () -> {};
        this.op = "";
        this.label = "";
    }
    
    /**
     * Creates a tensor with FP64 precision and gradient tracking enabled.
     */
    public Tensor(int rows, int cols, boolean requiresGrad) {
        this(rows, cols, Precision.FP64, requiresGrad);
    }
    
    /**
     * Creates a tensor with FP64 precision and no gradient tracking.
     */
    public Tensor(int rows, int cols) {
        this(rows, cols, Precision.FP64, false);
    }
    
    // Factory methods
    
    /**
     * Creates a tensor from a 2D array.
     * 
     * @param array 2D array of values (must be rectangular)
     * @param requiresGrad whether to track gradients
     * @return new tensor
     * @throws IllegalArgumentException if array is null, empty, or not rectangular
     */
    public static Tensor fromArray(double[][] array, boolean requiresGrad) {
        if (array == null || array.length == 0 || array[0].length == 0) {
            throw new IllegalArgumentException("Array must be non-empty");
        }
        
        int rows = array.length;
        int cols = array[0].length;
        
        // Validate rectangular
        for (int i = 1; i < rows; i++) {
            if (array[i].length != cols) {
                throw new IllegalArgumentException(
                    String.format("Array must be rectangular, row 0 has %d cols but row %d has %d cols",
                        cols, i, array[i].length));
            }
        }
        
        Tensor t = new Tensor(rows, cols, requiresGrad);
        for (int r = 0; r < rows; r++) {
            System.arraycopy(array[r], 0, t.data, r * cols, cols);
        }
        return t;
    }
    
    /**
     * Creates a tensor filled with zeros.
     */
    public static Tensor zeros(int rows, int cols, boolean requiresGrad) {
        return new Tensor(rows, cols, requiresGrad);
    }
    
    /**
     * Creates a tensor filled with ones.
     */
    public static Tensor ones(int rows, int cols, boolean requiresGrad) {
        Tensor t = new Tensor(rows, cols, requiresGrad);
        Arrays.fill(t.data, 1.0);
        return t;
    }
    
    /**
     * Creates a tensor filled with random values from a uniform distribution.
     * 
     * @param rows number of rows
     * @param cols number of columns
     * @param rng random generator
     * @param low lower bound (inclusive)
     * @param high upper bound (exclusive)
     * @param requiresGrad whether to track gradients
     * @return new tensor with random values
     */
    public static Tensor rand(int rows, int cols, RandomGenerator rng, double low, double high, boolean requiresGrad) {
        Tensor t = new Tensor(rows, cols, requiresGrad);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = low + (high - low) * rng.nextDouble();
        }
        return t;
    }
    
    // Accessors
    
    public int rows() {
        return rows;
    }
    
    public int cols() {
        return cols;
    }
    
    public int[] shape() {
        return new int[]{rows, cols};
    }
    
    public boolean isScalar() {
        return rows == 1 && cols == 1;
    }
    
    public double[] data() {
        return data;
    }
    
    public double[] grad() {
        if (!requiresGrad) {
            throw new IllegalStateException("Tensor does not require gradients");
        }
        return grad;
    }
    
    public boolean requiresGrad() {
        return requiresGrad;
    }
    
    public Precision precision() {
        return precision;
    }
    
    public String op() {
        return op;
    }
    
    public String label() {
        return label;
    }
    
    public void setLabel(String label) {
        this.label = label;
    }
    
    public Tensor[] getParents() {
        return parents;
    }
    
    /**
     * Gets element at (row, col).
     */
    public double get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException(
                String.format("Index (%d, %d) out of bounds for shape (%d, %d)", row, col, rows, cols));
        }
        return data[idx(row, col)];
    }
    
    /**
     * Sets element at (row, col).
     */
    public void set(int row, int col, double value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException(
                String.format("Index (%d, %d) out of bounds for shape (%d, %d)", row, col, rows, cols));
        }
        data[idx(row, col)] = value;
    }
    
    /**
     * Converts (row, col) to flat index.
     */
    private int idx(int row, int col) {
        return row * cols + col;
    }
    
    /**
     * Zeros out all gradients.
     */
    public void zeroGrad() {
        if (requiresGrad && grad != null) {
            Arrays.fill(grad, 0.0);
        }
    }
    
    // Autograd graph wiring
    
    void setParents(Tensor... parents) {
        this.parents = parents;
    }
    
    void setBackwardFn(Runnable fn) {
        this.backwardFn = fn;
    }
    
    void setOp(String op) {
        this.op = op;
    }
    
    // Backpropagation
    
    /**
     * Performs backpropagation from this tensor (typically a scalar loss).
     * 
     * <p>This method:
     * <ol>
     *   <li>Builds a topological ordering of the computation graph</li>
     *   <li>Zeros all gradients</li>
     *   <li>Sets this tensor's gradient to 1.0</li>
     *   <li>Traverses the graph in reverse order, calling backward functions</li>
     * </ol>
     * 
     * @throws IllegalStateException if this tensor is not a scalar
     */
    public static void backward(Tensor loss) {
        if (!loss.isScalar()) {
            throw new IllegalStateException(
                String.format("backward() requires a scalar tensor, got shape (%d, %d)", 
                    loss.rows, loss.cols));
        }
        
        if (!loss.requiresGrad) {
            throw new IllegalStateException("Loss tensor does not require gradients");
        }
        
        // Build topological order
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(loss, topo, visited);
        
        // Zero all gradients
        for (Tensor t : topo) {
            if (t.requiresGrad) {
                t.zeroGrad();
            }
        }
        
        // Seed gradient
        loss.grad[0] = 1.0;
        
        // Backpropagate
        for (int i = topo.size() - 1; i >= 0; i--) {
            Tensor t = topo.get(i);
            if (t.backwardFn != null) {
                t.backwardFn.run();
            }
        }
    }
    
    private static void buildTopo(Tensor t, List<Tensor> topo, Set<Tensor> visited) {
        if (visited.contains(t)) {
            return;
        }
        visited.add(t);
        for (Tensor parent : t.parents) {
            buildTopo(parent, topo, visited);
        }
        topo.add(t);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=(").append(rows).append(", ").append(cols).append(")");
        if (label != null && !label.isEmpty()) {
            sb.append(", label='").append(label).append("'");
        }
        if (!op.isEmpty()) {
            sb.append(", op='").append(op).append("'");
        }
        sb.append(", requiresGrad=").append(requiresGrad);
        sb.append(")");
        return sb.toString();
    }
    
    /**
     * Returns a string representation of the tensor data (for debugging).
     * Only shows first few elements if tensor is large.
     */
    public String dataString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int maxRows = Math.min(rows, 4);
        int maxCols = Math.min(cols, 4);
        
        for (int r = 0; r < maxRows; r++) {
            if (r > 0) sb.append(" ");
            sb.append("[");
            for (int c = 0; c < maxCols; c++) {
                if (c > 0) sb.append(", ");
                sb.append(String.format("%.4f", get(r, c)));
            }
            if (cols > maxCols) sb.append(", ...");
            sb.append("]");
            if (r < maxRows - 1) sb.append("\n");
        }
        if (rows > maxRows) sb.append("\n ...");
        sb.append("]");
        return sb.toString();
    }
}

