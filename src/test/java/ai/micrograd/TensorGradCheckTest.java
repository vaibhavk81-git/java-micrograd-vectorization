package ai.micrograd;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Gradient checking tests using central difference approximation.
 * 
 * <p>For each operation, we verify that the computed gradients match
 * numerical gradients within a relative error threshold.
 */
class TensorGradCheckTest {
    
    private static final double EPSILON = 1e-5;
    private static final double REL_ERROR_THRESHOLD = 1e-4;
    
    @Test
    void testAddGradient() {
        Tensor a = Tensor.fromArray(new double[][]{{1.0, 2.0}, {3.0, 4.0}}, true);
        Tensor b = Tensor.fromArray(new double[][]{{0.5, 1.5}, {2.5, 3.5}}, true);
        
        Tensor c = TensorOps.add(a, b);
        Tensor loss = TensorOps.sum(c, 0);
        loss = TensorOps.sum(loss, 1);  // Reduce to scalar
        
        Tensor.backward(loss);
        
        // Check gradients numerically
        checkGradient(a, () -> {
            Tensor c2 = TensorOps.add(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).item();
        });
        
        checkGradient(b, () -> {
            Tensor c2 = TensorOps.add(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).item();
        });
    }
    
    @Test
    void testMulGradient() {
        Tensor a = Tensor.fromArray(new double[][]{{1.0, 2.0}, {3.0, 4.0}}, true);
        Tensor b = Tensor.fromArray(new double[][]{{0.5, 1.5}, {2.5, 3.5}}, true);
        
        Tensor c = TensorOps.mul(a, b);
        Tensor loss = TensorOps.sum(c, 0);
        loss = TensorOps.sum(loss, 1);
        
        Tensor.backward(loss);
        
        checkGradient(a, () -> {
            Tensor c2 = TensorOps.mul(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).item();
        });
        
        checkGradient(b, () -> {
            Tensor c2 = TensorOps.mul(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).item();
        });
    }
    
    @Test
    void testTanhGradient() {
        Tensor x = Tensor.fromArray(new double[][]{{-1.0, 0.0}, {0.5, 1.0}}, true);
        
        Tensor y = TensorOps.tanh(x);
        Tensor loss = TensorOps.sum(y, 0);
        loss = TensorOps.sum(loss, 1);
        
        Tensor.backward(loss);
        
        checkGradient(x, () -> {
            Tensor y2 = TensorOps.tanh(x);
            Tensor l2 = TensorOps.sum(y2, 0);
            return TensorOps.sum(l2, 1).item();
        });
    }
    
    @Test
    void testReluGradient() {
        Tensor x = Tensor.fromArray(new double[][]{{-1.0, 0.5}, {1.0, 2.0}}, true);
        
        Tensor y = TensorOps.relu(x);
        Tensor loss = TensorOps.sum(y, 0);
        loss = TensorOps.sum(loss, 1);
        
        Tensor.backward(loss);
        
        checkGradient(x, () -> {
            Tensor y2 = TensorOps.relu(x);
            Tensor l2 = TensorOps.sum(y2, 0);
            return TensorOps.sum(l2, 1).item();
        });
    }
    
    @Test
    void testMatmulGradient() {
        Tensor a = Tensor.fromArray(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, true);
        Tensor b = Tensor.fromArray(new double[][]{{0.5, 1.0}, {1.5, 2.0}, {2.5, 3.0}}, true);
        
        Tensor c = TensorOps.matmul(a, b);
        Tensor loss = TensorOps.sum(c, 0);
        loss = TensorOps.sum(loss, 1);
        
        Tensor.backward(loss);
        
        checkGradient(a, () -> {
            Tensor c2 = TensorOps.matmul(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).item();
        });
        
        checkGradient(b, () -> {
            Tensor c2 = TensorOps.matmul(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).item();
        });
    }
    
    @Test
    void testSumAxis0Gradient() {
        Tensor x = Tensor.fromArray(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, true);
        
        Tensor y = TensorOps.sum(x, 0);  // (1×3)
        Tensor loss = TensorOps.sum(y, 1);  // scalar
        
        Tensor.backward(loss);
        
        checkGradient(x, () -> {
            Tensor y2 = TensorOps.sum(x, 0);
            return TensorOps.sum(y2, 1).item();
        });
    }
    
    @Test
    void testSumAxis1Gradient() {
        Tensor x = Tensor.fromArray(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, true);
        
        Tensor y = TensorOps.sum(x, 1);  // (2×1)
        Tensor loss = TensorOps.sum(y, 0);  // scalar
        
        Tensor.backward(loss);
        
        checkGradient(x, () -> {
            Tensor y2 = TensorOps.sum(x, 1);
            return TensorOps.sum(y2, 0).item();
        });
    }
    
    @Test
    void testMeanAxis0Gradient() {
        Tensor x = Tensor.fromArray(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, true);
        
        Tensor y = TensorOps.mean(x, 0);  // (1×3)
        Tensor loss = TensorOps.sum(y, 1);  // scalar
        
        Tensor.backward(loss);
        
        checkGradient(x, () -> {
            Tensor y2 = TensorOps.mean(x, 0);
            return TensorOps.sum(y2, 1).item();
        });
    }
    
    @Test
    void testMeanAxis1Gradient() {
        Tensor x = Tensor.fromArray(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, true);
        
        Tensor y = TensorOps.mean(x, 1);  // (2×1)
        Tensor loss = TensorOps.sum(y, 0);  // scalar
        
        Tensor.backward(loss);
        
        checkGradient(x, () -> {
            Tensor y2 = TensorOps.mean(x, 1);
            return TensorOps.sum(y2, 0).item();
        });
    }
    
    /**
     * Checks gradient using central difference approximation.
     * 
     * @param tensor tensor to check gradients for
     * @param lossFunc function that computes scalar loss
     */
    private void checkGradient(Tensor tensor, java.util.function.Supplier<Double> lossFunc) {
        int elements = tensor.elements();
        
        for (int i = 0; i < elements; i++) {
            int row = i / tensor.cols();
            int col = i % tensor.cols();
            double original = tensor.get(row, col);
            
            tensor.set(row, col, original + EPSILON);
            double lossPlus = lossFunc.get();
            
            tensor.set(row, col, original - EPSILON);
            double lossMinus = lossFunc.get();
            
            tensor.set(row, col, original);  // Restore
            
            double numericalGrad = (lossPlus - lossMinus) / (2 * EPSILON);
            double analyticalGrad = tensor.gradAt(i);
            
            // Compute relative error
            double relError = Math.abs(numericalGrad - analyticalGrad) / 
                             (Math.abs(numericalGrad) + Math.abs(analyticalGrad) + 1e-8);
            
            assertTrue(relError < REL_ERROR_THRESHOLD,
                String.format("Gradient mismatch at index %d: numerical=%.6f, analytical=%.6f, relError=%.6f",
                    i, numericalGrad, analyticalGrad, relError));
        }
    }
}

