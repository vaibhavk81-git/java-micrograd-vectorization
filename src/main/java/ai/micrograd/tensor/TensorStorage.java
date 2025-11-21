package ai.micrograd.tensor;

import java.util.random.RandomGenerator;

/**
 * Abstraction over tensor data/gradient buffers.
 *
 * <p>Implementations own the actual memory (e.g., JVM heap arrays for CPU,
 * direct memory or device allocations for future GPUs) and provide scalar-level
 * operations that higher-level tensor logic can rely on without knowing the
 * underlying representation.</p>
 */
interface TensorStorage {

    DeviceType device();

    Precision precision();

    /**
     * Number of scalar elements in the tensor (rows * cols).
     */
    int size();

    /**
     * Whether gradients are materialized for this storage block.
     */
    boolean requiresGrad();

    /**
     * Fills the data buffer with a constant value (converted to the native
     * precision internally).
     */
    void fill(double value);

    /**
     * Copies values from a flattened double array into the storage buffer.
     * Length must match {@link #size()}.
     */
    void copyFrom(double[] values);

    /**
     * Samples uniformly from {@code [low, high)} into the data buffer.
     */
    void randomUniform(RandomGenerator rng, double low, double high);

    /**
     * Reads a single element (row-major index) as double precision.
     */
    double read(int index);

    /**
     * Writes a single element (row-major index) converting from double to the
     * storage precision if necessary.
     */
    void write(int index, double value);

    /**
     * Zeros the gradient buffer (no-op if gradients are not tracked).
     */
    void zeroGrad();

    /**
     * Fills the gradient buffer with a constant value.
     */
    void fillGrad(double value);

    /**
     * Writes a gradient value at {@code index}.
     */
    void writeGrad(int index, double value);

    /**
     * Reads a gradient value at {@code index} as double precision.
     */
    double readGrad(int index);

    /**
     * Returns a defensive copy of the data buffer as double precision values.
     */
    double[] toDoubleArray();

    /**
     * Returns a defensive copy of the gradient buffer (requires gradients).
     */
    double[] gradToDoubleArray();
}


