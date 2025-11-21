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
            return TensorOps.sum(l2, 1).data()[0];
        });
        
        checkGradient(b, () -> {
            Tensor c2 = TensorOps.add(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).data()[0];
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
            return TensorOps.sum(l2, 1).data()[0];
        });
        
        checkGradient(b, () -> {
            Tensor c2 = TensorOps.mul(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).data()[0];
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
            return TensorOps.sum(l2, 1).data()[0];
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
            return TensorOps.sum(l2, 1).data()[0];
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
            return TensorOps.sum(l2, 1).data()[0];
        });
        
        checkGradient(b, () -> {
            Tensor c2 = TensorOps.matmul(a, b);
            Tensor l2 = TensorOps.sum(c2, 0);
            return TensorOps.sum(l2, 1).data()[0];
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
            return TensorOps.sum(y2, 1).data()[0];
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
            return TensorOps.sum(y2, 0).data()[0];
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
            return TensorOps.sum(y2, 1).data()[0];
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
            return TensorOps.sum(y2, 0).data()[0];
        });
    }
    
    /**
     * Checks gradient using central difference approximation.
     * 
     * @param tensor tensor to check gradients for
     * @param lossFunc function that computes scalar loss
     */
    private void checkGradient(Tensor tensor, java.util.function.Supplier<Double> lossFunc) {
        double[] data = tensor.data();
        double[] grad = tensor.grad();
        
        for (int i = 0; i < data.length; i++) {
            // Compute numerical gradient using central difference
            double original = data[i];
            
            data[i] = original + EPSILON;
            double lossPlus = lossFunc.get();
            
            data[i] = original - EPSILON;
            double lossMinus = lossFunc.get();
            
            data[i] = original;  // Restore
            
            double numericalGrad = (lossPlus - lossMinus) / (2 * EPSILON);
            double analyticalGrad = grad[i];
            
            // Compute relative error
            double relError = Math.abs(numericalGrad - analyticalGrad) / 
                             (Math.abs(numericalGrad) + Math.abs(analyticalGrad) + 1e-8);
            
            assertTrue(relError < REL_ERROR_THRESHOLD,
                String.format("Gradient mismatch at index %d: numerical=%.6f, analytical=%.6f, relError=%.6f",
                    i, numericalGrad, analyticalGrad, relError));
        }
    }
}

