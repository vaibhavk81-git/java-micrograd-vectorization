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
     * Currently the only fully supported mode.
     */
    FP64,
    
    /**
     * 32-bit floating point (single precision).
     * Currently not implemented - throws UnsupportedOperationException.
     * Reserved for future GPU/performance optimizations.
     */
    FP32
}

