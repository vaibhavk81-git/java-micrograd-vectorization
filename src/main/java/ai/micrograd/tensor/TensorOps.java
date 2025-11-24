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

    private TensorOps() {}

    public static Tensor add(Tensor a, Tensor b) {
        ensureSameDevice(a, b, "add");
        return backendFor(a).add(a, b);
    }

    public static Tensor mul(Tensor a, Tensor b) {
        ensureSameDevice(a, b, "mul");
        return backendFor(a).mul(a, b);
    }

    public static Tensor tanh(Tensor x) {
        return backendFor(x).tanh(x);
    }

    public static Tensor relu(Tensor x) {
        return backendFor(x).relu(x);
    }

    public static Tensor addRowVector(Tensor matrix, Tensor rowVec) {
        ensureSameDevice(matrix, rowVec, "addRowVector");
        return backendFor(matrix).addRowVector(matrix, rowVec);
    }

    public static Tensor sum(Tensor x, int axis) {
        return backendFor(x).sum(x, axis);
    }

    public static Tensor mean(Tensor x, int axis) {
        return backendFor(x).mean(x, axis);
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        ensureSameDevice(a, b, "matmul");
        return backendFor(a).matmul(a, b);
    }

    /**
     * PyTorch equivalent: {@code torch.nn.functional.embedding}.
     * Returns the gathered embedding rows for the provided indices.
     */
    public static Tensor embedding(Tensor weight, Tensor indices) {
        return backendFor(weight).embedding(weight, indices);
    }

    /**
     * PyTorch equivalent: {@code torch.nn.functional.cross_entropy}.
     * Computes log-softmax + NLL loss (mean over batch).
     */
    public static Tensor crossEntropy(Tensor logits, Tensor targets) {
        return backendFor(logits).crossEntropy(logits, targets);
    }

    /**
     * PyTorch equivalent: {@code tensor.view(newRows, newCols)}.
     */
    public static Tensor reshape(Tensor input, int newRows, int newCols) {
        return backendFor(input).reshape(input, newRows, newCols);
    }

    private static TensorBackend backendFor(Tensor tensor) {
        return TensorBackendRegistry.get(tensor.device());
    }

    private static void ensureSameDevice(Tensor a, Tensor b, String op) {
        if (a.device() != b.device()) {
            throw new IllegalArgumentException(
                String.format("%s requires tensors on the same device, got %s vs %s",
                    op, a.device(), b.device()));
        }
    }
}

