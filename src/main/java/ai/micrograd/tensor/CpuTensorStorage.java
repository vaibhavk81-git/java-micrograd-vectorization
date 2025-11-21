package ai.micrograd.tensor;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * CPU-backed tensor storage that keeps data/gradient buffers on the JVM heap.
 *
 * <p>Depending on {@link Precision}, the buffers are either {@code double[]}
 * (FP64) or {@code float[]} (FP32). All public methods convert to/from
 * {@code double} so that higher levels can remain precision-agnostic.</p>
 */
final class CpuTensorStorage implements TensorStorage {

    private final Precision precision;
    private final int size;
    private final boolean requiresGrad;
    private final double[] data64;
    private final float[] data32;
    private final double[] grad64;
    private final float[] grad32;

    CpuTensorStorage(int size, Precision precision, boolean requiresGrad) {
        if (size <= 0) {
            throw new IllegalArgumentException("Storage size must be positive, got: " + size);
        }
        this.precision = precision;
        this.size = size;
        this.requiresGrad = requiresGrad;

        if (precision == Precision.FP64) {
            this.data64 = new double[size];
            this.grad64 = requiresGrad ? new double[size] : null;
            this.data32 = null;
            this.grad32 = null;
        } else {
            this.data64 = null;
            this.grad64 = null;
            this.data32 = new float[size];
            this.grad32 = requiresGrad ? new float[size] : null;
        }
    }

    @Override
    public DeviceType device() {
        return DeviceType.CPU;
    }

    @Override
    public Precision precision() {
        return precision;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean requiresGrad() {
        return requiresGrad;
    }

    @Override
    public void fill(double value) {
        if (precision == Precision.FP64) {
            Arrays.fill(data64, value);
        } else {
            Arrays.fill(data32, (float) value);
        }
    }

    @Override
    public void copyFrom(double[] values) {
        if (values.length != size) {
            throw new IllegalArgumentException(
                "copyFrom length mismatch: expected " + size + ", got " + values.length);
        }
        if (precision == Precision.FP64) {
            System.arraycopy(values, 0, data64, 0, size);
        } else {
            for (int i = 0; i < size; i++) {
                data32[i] = (float) values[i];
            }
        }
    }

    @Override
    public void randomUniform(RandomGenerator rng, double low, double high) {
        if (precision == Precision.FP64) {
            for (int i = 0; i < size; i++) {
                data64[i] = low + (high - low) * rng.nextDouble();
            }
        } else {
            for (int i = 0; i < size; i++) {
                data32[i] = (float) (low + (high - low) * rng.nextDouble());
            }
        }
    }

    @Override
    public double read(int index) {
        validateIndex(index);
        return precision == Precision.FP64 ? data64[index] : data32[index];
    }

    @Override
    public void write(int index, double value) {
        validateIndex(index);
        if (precision == Precision.FP64) {
            data64[index] = value;
        } else {
            data32[index] = (float) value;
        }
    }

    @Override
    public void zeroGrad() {
        if (!requiresGrad || (grad64 == null && grad32 == null)) {
            return;
        }
        if (precision == Precision.FP64) {
            Arrays.fill(grad64, 0.0);
        } else {
            Arrays.fill(grad32, 0.0f);
        }
    }

    @Override
    public void fillGrad(double value) {
        ensureGrad("fillGrad");
        if (precision == Precision.FP64) {
            Arrays.fill(grad64, value);
        } else {
            Arrays.fill(grad32, (float) value);
        }
    }

    @Override
    public void writeGrad(int index, double value) {
        ensureGrad("writeGrad");
        validateIndex(index);
        if (precision == Precision.FP64) {
            grad64[index] = value;
        } else {
            grad32[index] = (float) value;
        }
    }

    @Override
    public double readGrad(int index) {
        ensureGrad("readGrad");
        validateIndex(index);
        return precision == Precision.FP64 ? grad64[index] : grad32[index];
    }

    @Override
    public double[] toDoubleArray() {
        if (precision == Precision.FP64) {
            return Arrays.copyOf(data64, data64.length);
        }
        double[] copy = new double[size];
        for (int i = 0; i < size; i++) {
            copy[i] = data32[i];
        }
        return copy;
    }

    @Override
    public double[] gradToDoubleArray() {
        ensureGrad("gradToDoubleArray");
        if (precision == Precision.FP64) {
            return Arrays.copyOf(grad64, grad64.length);
        }
        double[] copy = new double[size];
        for (int i = 0; i < size; i++) {
            copy[i] = grad32[i];
        }
        return copy;
    }

    double[] data64() {
        if (precision != Precision.FP64) {
            throw new IllegalStateException("Storage is not FP64");
        }
        return data64;
    }

    float[] data32() {
        if (precision != Precision.FP32) {
            throw new IllegalStateException("Storage is not FP32");
        }
        return data32;
    }

    double[] grad64() {
        if (precision != Precision.FP64) {
            throw new IllegalStateException("Storage is not FP64");
        }
        if (grad64 == null) {
            throw new IllegalStateException("Gradients not enabled for this tensor");
        }
        return grad64;
    }

    float[] grad32() {
        if (precision != Precision.FP32) {
            throw new IllegalStateException("Storage is not FP32");
        }
        if (grad32 == null) {
            throw new IllegalStateException("Gradients not enabled for this tensor");
        }
        return grad32;
    }

    private void validateIndex(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for size " + size);
        }
    }

    private void ensureGrad(String op) {
        if (!requiresGrad || (grad64 == null && grad32 == null)) {
            throw new IllegalStateException(op + " requires gradients to be enabled");
        }
    }
}


