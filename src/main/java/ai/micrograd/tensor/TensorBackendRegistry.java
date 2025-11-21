package ai.micrograd.tensor;

import java.util.EnumMap;
import java.util.Map;

/**
 * Central registry that maps {@link DeviceType} to the corresponding backend
 * implementation. Today only CPU is available, but the indirection keeps the
 * rest of the codebase clean and future-ready.
 */
final class TensorBackendRegistry {

    private static final Map<DeviceType, TensorBackend> BACKENDS = new EnumMap<>(DeviceType.class);

    static {
        TensorBackend cpu = new CpuBackend();
        BACKENDS.put(cpu.deviceType(), cpu);
    }

    private TensorBackendRegistry() {}

    static TensorBackend get(DeviceType device) {
        TensorBackend backend = BACKENDS.get(device);
        if (backend == null) {
            throw new UnsupportedOperationException("No backend registered for device: " + device);
        }
        return backend;
    }
}


