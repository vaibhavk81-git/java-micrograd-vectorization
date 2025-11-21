package ai.micrograd.tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;

/**
 * Tensor represents a 2-D dense tensor with automatic differentiation capabilities.
 *
 * <p><b>Storage:</b> Data lives inside a {@link TensorStorage} instance so the
 * rest of the system can stay agnostic to whether we're using doubles, floats,
 * or future GPU buffers.</p>
 */
public final class Tensor {

    private static final DeviceType DEFAULT_DEVICE = DeviceManager.get().defaultDevice();
    private static Precision defaultPrecision = Precision.FP64;

    public static Precision getDefaultPrecision() {
        return defaultPrecision;
    }

    public static void setDefaultPrecision(Precision precision) {
        if (precision == null) {
            throw new IllegalArgumentException("Default precision cannot be null");
        }
        defaultPrecision = precision;
    }

    private final int rows;
    private final int cols;
    private final boolean requiresGrad;
    private final Precision precision;
    private final DeviceType device;
    private final TensorBackend backend;
    private final TensorStorage storage;

    // Autograd graph
    private Tensor[] parents;
    private Runnable backwardFn;
    private String op;
    private String label;

    public Tensor(int rows, int cols, Precision precision, boolean requiresGrad) {
        this(rows, cols, precision, DEFAULT_DEVICE, requiresGrad);
    }

    public Tensor(int rows, int cols, Precision precision, DeviceType device, boolean requiresGrad) {
        if (rows < 1 || cols < 1) {
            throw new IllegalArgumentException(
                String.format("Tensor dimensions must be positive, got: (%d, %d)", rows, cols));
        }
        this.rows = rows;
        this.cols = cols;
        this.precision = precision;
        this.requiresGrad = requiresGrad;
        this.device = device;
        this.backend = TensorBackendRegistry.get(device);
        this.storage = backend.allocate(rows * cols, precision, requiresGrad);
        this.parents = new Tensor[0];
        this.backwardFn = () -> {};
        this.op = "";
        this.label = "";
    }

    public Tensor(int rows, int cols, boolean requiresGrad) {
        this(rows, cols, defaultPrecision, DEFAULT_DEVICE, requiresGrad);
    }

    public Tensor(int rows, int cols) {
        this(rows, cols, defaultPrecision, DEFAULT_DEVICE, false);
    }

    // Factory methods

    public static Tensor fromArray(double[][] array, boolean requiresGrad) {
        if (array == null || array.length == 0 || array[0].length == 0) {
            throw new IllegalArgumentException("Array must be non-empty");
        }

        int rows = array.length;
        int cols = array[0].length;

        for (int i = 1; i < rows; i++) {
            if (array[i].length != cols) {
                throw new IllegalArgumentException(
                    String.format("Array must be rectangular, row 0 has %d cols but row %d has %d cols",
                        cols, i, array[i].length));
            }
        }

        Tensor t = new Tensor(rows, cols, requiresGrad);
        double[] flattened = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(array[r], 0, flattened, r * cols, cols);
        }
        t.storage.copyFrom(flattened);
        return t;
    }

    public static Tensor zeros(int rows, int cols, boolean requiresGrad) {
        return zeros(rows, cols, defaultPrecision, DEFAULT_DEVICE, requiresGrad);
    }

    public static Tensor zeros(int rows, int cols, Precision precision, boolean requiresGrad) {
        return zeros(rows, cols, precision, DEFAULT_DEVICE, requiresGrad);
    }

    public static Tensor zeros(int rows, int cols, Precision precision, DeviceType device, boolean requiresGrad) {
        return new Tensor(rows, cols, precision, device, requiresGrad);
    }

    public static Tensor ones(int rows, int cols, boolean requiresGrad) {
        return ones(rows, cols, defaultPrecision, DEFAULT_DEVICE, requiresGrad);
    }

    public static Tensor ones(int rows, int cols, Precision precision, boolean requiresGrad) {
        return ones(rows, cols, precision, DEFAULT_DEVICE, requiresGrad);
    }

    public static Tensor ones(int rows, int cols, Precision precision, DeviceType device, boolean requiresGrad) {
        Tensor t = zeros(rows, cols, precision, device, requiresGrad);
        t.fill(1.0);
        return t;
    }

    public static Tensor full(int rows, int cols, double value, boolean requiresGrad) {
        return full(rows, cols, value, defaultPrecision, DEFAULT_DEVICE, requiresGrad);
    }

    public static Tensor full(int rows, int cols, double value, Precision precision, boolean requiresGrad) {
        return full(rows, cols, value, precision, DEFAULT_DEVICE, requiresGrad);
    }

    public static Tensor full(int rows, int cols, double value, Precision precision, DeviceType device,
                              boolean requiresGrad) {
        Tensor t = new Tensor(rows, cols, precision, device, requiresGrad);
        t.fill(value);
        return t;
    }

    public static Tensor rand(int rows, int cols, RandomGenerator rng, double low, double high, boolean requiresGrad) {
        return rand(rows, cols, defaultPrecision, DEFAULT_DEVICE, rng, low, high, requiresGrad);
    }

    public static Tensor rand(int rows, int cols, Precision precision, RandomGenerator rng, double low, double high,
                              boolean requiresGrad) {
        return rand(rows, cols, precision, DEFAULT_DEVICE, rng, low, high, requiresGrad);
    }

    public static Tensor rand(int rows, int cols, Precision precision, DeviceType device, RandomGenerator rng,
                              double low, double high, boolean requiresGrad) {
        Tensor t = new Tensor(rows, cols, precision, device, requiresGrad);
        t.storage.randomUniform(rng, low, high);
        return t;
    }

    public static Tensor zerosLike(Tensor template) {
        return zerosLike(template, template.requiresGrad());
    }

    public static Tensor zerosLike(Tensor template, boolean requiresGrad) {
        return zeros(template.rows(), template.cols(), template.precision(), template.device(), requiresGrad);
    }

    public static Tensor onesLike(Tensor template) {
        return onesLike(template, template.requiresGrad());
    }

    public static Tensor onesLike(Tensor template, boolean requiresGrad) {
        return ones(template.rows(), template.cols(), template.precision(), template.device(), requiresGrad);
    }

    public static Tensor fullLike(Tensor template, double value) {
        return fullLike(template, value, template.requiresGrad());
    }

    public static Tensor fullLike(Tensor template, double value, boolean requiresGrad) {
        return full(template.rows(), template.cols(), value, template.precision(), template.device(), requiresGrad);
    }

    public static Tensor randLike(Tensor template, RandomGenerator rng, double low, double high) {
        return randLike(template, rng, low, high, template.requiresGrad());
    }

    public static Tensor randLike(Tensor template, RandomGenerator rng, double low, double high, boolean requiresGrad) {
        return rand(template.rows(), template.cols(), template.precision(), template.device(), rng, low, high,
            requiresGrad);
    }

    static Tensor emptyLike(Tensor template, boolean requiresGrad) {
        return new Tensor(template.rows, template.cols, template.precision, template.device, requiresGrad);
    }

    public static Tensor fromArray(double[][] array, Precision precision, boolean requiresGrad) {
        if (array == null || array.length == 0 || array[0].length == 0) {
            throw new IllegalArgumentException("Array must be non-empty");
        }

        int rows = array.length;
        int cols = array[0].length;

        for (int i = 1; i < rows; i++) {
            if (array[i].length != cols) {
                throw new IllegalArgumentException(
                    String.format("Array must be rectangular, row 0 has %d cols but row %d has %d cols",
                        cols, i, array[i].length));
            }
        }

        Tensor t = new Tensor(rows, cols, precision, requiresGrad);
        double[] flattened = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(array[r], 0, flattened, r * cols, cols);
        }
        t.storage.copyFrom(flattened);
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

    public int elements() {
        return rows * cols;
    }

    public boolean isScalar() {
        return rows == 1 && cols == 1;
    }

    public DeviceType device() {
        return device;
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

    public double get(int row, int col) {
        validateIndex(row, col);
        return storage.read(idx(row, col));
    }

    public void set(int row, int col, double value) {
        validateIndex(row, col);
        storage.write(idx(row, col), value);
    }

    public void fill(double value) {
        storage.fill(value);
    }

    public double item() {
        if (!isScalar()) {
            throw new IllegalStateException(
                String.format("item() requires a scalar tensor, got shape (%d, %d)", rows, cols));
        }
        return storage.read(0);
    }

    public double[] toArray() {
        return storage.toDoubleArray();
    }

    public double[] gradToArray() {
        ensureGradEnabled("gradToArray");
        return storage.gradToDoubleArray();
    }

    public Tensor to(DeviceType targetDevice) {
        return to(targetDevice, null);
    }

    public Tensor to(Precision targetPrecision) {
        return to(null, targetPrecision);
    }

    public Tensor to(DeviceType targetDevice, Precision targetPrecision) {
        DeviceType resolvedDevice = targetDevice != null ? targetDevice : this.device;
        Precision resolvedPrecision = targetPrecision != null ? targetPrecision : this.precision;
        if (resolvedDevice == this.device && resolvedPrecision == this.precision) {
            return this;
        }
        Tensor converted = new Tensor(rows, cols, resolvedPrecision, resolvedDevice, requiresGrad);
        converted.storage.copyFrom(this.toArray());
        if (requiresGrad) {
            double[] grads = this.gradToArray();
            for (int i = 0; i < grads.length; i++) {
                converted.storage.writeGrad(i, grads[i]);
            }
        }
        converted.label = this.label;
        converted.op = this.op;
        return converted;
    }

    public void fillGrad(double value) {
        ensureGradEnabled("fillGrad");
        storage.fillGrad(value);
    }

    public void setGradAt(int index, double value) {
        ensureGradEnabled("setGradAt");
        storage.writeGrad(index, value);
    }

    public double gradAt(int index) {
        ensureGradEnabled("gradAt");
        return storage.readGrad(index);
    }

    public void zeroGrad() {
        if (requiresGrad) {
            storage.zeroGrad();
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

    TensorStorage storage() {
        return storage;
    }

    double getValueFlat(int index) {
        return storage.read(index);
    }

    void accumulateGradAt(int index, double value) {
        ensureGradEnabled("accumulateGradAt");
        double current = storage.readGrad(index);
        storage.writeGrad(index, current + value);
    }

    // Backpropagation

    public static void backward(Tensor loss) {
        if (!loss.isScalar()) {
            throw new IllegalStateException(
                String.format("backward() requires a scalar tensor, got shape (%d, %d)",
                    loss.rows, loss.cols));
        }

        if (!loss.requiresGrad) {
            throw new IllegalStateException("Loss tensor does not require gradients");
        }

        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(loss, topo, visited);

        for (Tensor t : topo) {
            if (t.requiresGrad) {
                t.zeroGrad();
            }
        }

        loss.storage.writeGrad(0, 1.0);

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
        sb.append(", precision=").append(precision);
        sb.append(", device=").append(device.cliValue());
        sb.append(")");
        return sb.toString();
    }

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

    private int idx(int row, int col) {
        return row * cols + col;
    }

    private void validateIndex(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException(
                String.format("Index (%d, %d) out of bounds for shape (%d, %d)", row, col, rows, cols));
        }
    }

    private void ensureGradEnabled(String opName) {
        if (!requiresGrad) {
            throw new IllegalStateException(opName + " requires gradients to be enabled");
        }
    }
}

