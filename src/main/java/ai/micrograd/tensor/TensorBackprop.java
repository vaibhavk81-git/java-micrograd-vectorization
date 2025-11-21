package ai.micrograd.tensor;

/**
 * Legacy shim retained for teaching materials. All math kernels now live inside
 * {@link TensorBackend} implementations (currently {@link CpuBackend}). This
 * class simply exposes helpers to retrieve the active backend so historical
 * references keep compiling.
 */
final class TensorBackprop {

    private TensorBackprop() {}

    static TensorBackend backendFor(DeviceType device) {
        return TensorBackendRegistry.get(device);
    }

    static TensorBackend cpu() {
        return TensorBackendRegistry.get(DeviceType.CPU);
    }
}

