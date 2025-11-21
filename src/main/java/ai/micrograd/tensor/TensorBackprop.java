package ai.micrograd.tensor;

/**
 * TensorBackprop provides helper utilities for gradient computation.
 * 
 * <p>These are package-private helpers used by TensorOps to reduce code duplication
 * and improve performance in gradient computations.
 * 
 * @author Vaibhav Khare
 */
final class TensorBackprop {
    
    private TensorBackprop() {} // Prevent instantiation
    
    /**
     * Transposes a matrix in-place into a destination array.
     * 
     * <p>Useful for matmul gradients where we need A^T or B^T.
     * 
     * @param src source data (row-major, rows×cols)
     * @param rows number of rows in source
     * @param cols number of columns in source
     * @param dst destination array (must be cols×rows in size)
     */
    static void transpose(double[] src, int rows, int cols, double[] dst) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
    }
    
    /**
     * Matrix multiplication into a destination array (no allocation).
     * 
     * <p>Computes: dst += a @ b
     * 
     * @param a left matrix data (m×k)
     * @param aRows rows of a
     * @param aCols columns of a (= rows of b)
     * @param b right matrix data (k×n)
     * @param bCols columns of b
     * @param dst destination array (m×n), gradients accumulated here
     */
    static void matmulInto(double[] a, int aRows, int aCols, double[] b, int bCols, double[] dst) {
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                double sum = 0.0;
                for (int k = 0; k < aCols; k++) {
                    sum += a[i * aCols + k] * b[k * bCols + j];
                }
                dst[i * bCols + j] += sum;
            }
        }
    }
    
    /**
     * Broadcasts a row vector gradient back to matrix gradient by summing over rows.
     * 
     * <p>Used in addRowVector backward pass.
     * 
     * @param matrixGrad upstream gradient (m×n)
     * @param rows number of rows
     * @param cols number of columns
     * @param rowVecGrad destination for row vector gradient (1×n)
     */
    static void broadcastGradRow(double[] matrixGrad, int rows, int cols, double[] rowVecGrad) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                rowVecGrad[c] += matrixGrad[r * cols + c];
            }
        }
    }
    
    /**
     * Expands a reduced gradient back to full shape.
     * 
     * <p>Used in sum/mean backward pass.
     * 
     * @param reducedGrad gradient of reduced tensor
     * @param axis axis that was reduced (0 or 1)
     * @param fullRows rows of full tensor
     * @param fullCols cols of full tensor
     * @param fullGrad destination for full gradient
     * @param scale scaling factor (1.0 for sum, 1/count for mean)
     */
    static void reduceExpand(double[] reducedGrad, int axis, int fullRows, int fullCols, 
                             double[] fullGrad, double scale) {
        if (axis == 0) {
            // Expand (1×n) -> (m×n)
            for (int r = 0; r < fullRows; r++) {
                for (int c = 0; c < fullCols; c++) {
                    fullGrad[r * fullCols + c] += reducedGrad[c] * scale;
                }
            }
        } else {
            // Expand (m×1) -> (m×n)
            for (int r = 0; r < fullRows; r++) {
                for (int c = 0; c < fullCols; c++) {
                    fullGrad[r * fullCols + c] += reducedGrad[r] * scale;
                }
            }
        }
    }
}

