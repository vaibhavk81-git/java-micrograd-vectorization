package ai.micrograd.tensor;

/**
 * Small helper exposing backend lookups to code outside the tensor package.
 */
public final class TensorRuntime {

    private TensorRuntime() {}

    public static TensorBackend backendFor(Tensor tensor) {
        return TensorBackendRegistry.get(tensor.device());
    }
}

