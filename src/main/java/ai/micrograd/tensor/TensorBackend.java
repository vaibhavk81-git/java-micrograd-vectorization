package ai.micrograd.tensor;

/**
 * Backend responsible for executing tensor math on a specific device.
 *
 * <p>The backend owns the tight math loops so that higher layers stay agnostic
 * to how data is laid out (double arrays, float arrays, GPU buffers, etc.).</p>
 */
public interface TensorBackend {

    DeviceType deviceType();

    /**
     * Allocates a storage block for {@code size} elements.
     */
    TensorStorage allocate(int size, Precision precision, boolean requiresGrad);

    Tensor add(Tensor a, Tensor b);

    Tensor mul(Tensor a, Tensor b);

    Tensor tanh(Tensor x);

    Tensor relu(Tensor x);

    Tensor addRowVector(Tensor matrix, Tensor rowVec);

    Tensor sum(Tensor x, int axis);

    Tensor mean(Tensor x, int axis);

    Tensor matmul(Tensor a, Tensor b);

    /**
     * In-place SGD update (data -= lr * grad).
     */
    void sgdStep(Tensor param, double lr);
}


