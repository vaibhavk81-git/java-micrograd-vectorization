package ai.micrograd.tensor;

/**
 * Precision mode for tensor computations.
 * 
 * <p>This enum allows switching between different floating-point precisions
 * for future optimization and GPU acceleration support.
 * 
 * @author Vaibhav Khare
 */
public enum Precision {
    /**
     * 64-bit floating point (double precision).
     */
    FP64(64, "float64"),
    
    /**
     * 32-bit floating point (single precision).
     */
    FP32(32, "float32");

    private final int bits;
    private final String displayName;

    Precision(int bits, String displayName) {
        this.bits = bits;
        this.displayName = displayName;
    }

    public int bits() {
        return bits;
    }

    public String displayName() {
        return displayName;
    }
}

