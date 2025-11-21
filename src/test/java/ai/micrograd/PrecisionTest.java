package ai.micrograd;

import ai.micrograd.tensor.Precision;
import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;
import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for FP32 and FP64 precision modes.
 */
class PrecisionTest {

    @Test
    void testFP64TensorCreation() {
        Tensor t = new Tensor(2, 3, Precision.FP64, false);
        assertEquals(Precision.FP64, t.precision());
        assertEquals(2, t.rows());
        assertEquals(3, t.cols());
    }

    @Test
    void testFP32TensorCreation() {
        Tensor t = new Tensor(2, 3, Precision.FP32, false);
        assertEquals(Precision.FP32, t.precision());
        assertEquals(2, t.rows());
        assertEquals(3, t.cols());
    }

    @Test
    void testFP32BasicOperations() {
        Tensor a = Tensor.ones(2, 2, Precision.FP32, false);
        Tensor b = Tensor.ones(2, 2, Precision.FP32, false);
        
        Tensor c = TensorOps.add(a, b);
        assertEquals(Precision.FP32, c.precision());
        assertEquals(2.0, c.get(0, 0), 1e-6);
        assertEquals(2.0, c.get(1, 1), 1e-6);
    }

    @Test
    void testFP32Multiplication() {
        Tensor a = Tensor.full(2, 2, 3.0, Precision.FP32, false);
        Tensor b = Tensor.full(2, 2, 2.0, Precision.FP32, false);
        
        Tensor c = TensorOps.mul(a, b);
        assertEquals(Precision.FP32, c.precision());
        assertEquals(6.0, c.get(0, 0), 1e-5);
        assertEquals(6.0, c.get(1, 1), 1e-5);
    }

    @Test
    void testFP32Matmul() {
        Tensor a = Tensor.ones(2, 3, Precision.FP32, false);
        Tensor b = Tensor.ones(3, 2, Precision.FP32, false);
        
        Tensor c = TensorOps.matmul(a, b);
        assertEquals(Precision.FP32, c.precision());
        assertEquals(2, c.rows());
        assertEquals(2, c.cols());
        // Each element should be sum of 3 ones = 3.0
        assertEquals(3.0, c.get(0, 0), 1e-5);
        assertEquals(3.0, c.get(1, 1), 1e-5);
    }

    @Test
    void testFP32Activations() {
        Tensor x = Tensor.fromArray(new double[][]{{-1.0, 0.0}, {0.5, 1.0}}, Precision.FP32, false);
        
        Tensor tanh = TensorOps.tanh(x);
        assertEquals(Precision.FP32, tanh.precision());
        assertTrue(tanh.get(0, 0) < 0);  // tanh(-1) < 0
        assertEquals(0.0, tanh.get(0, 1), 1e-5);  // tanh(0) = 0
        
        Tensor relu = TensorOps.relu(x);
        assertEquals(Precision.FP32, relu.precision());
        assertEquals(0.0, relu.get(0, 0), 1e-5);  // relu(-1) = 0
        assertEquals(0.5, relu.get(1, 0), 1e-5);  // relu(0.5) = 0.5
    }

    @Test
    void testFP32Gradients() {
        Tensor a = Tensor.fromArray(new double[][]{{1.0, 2.0}}, Precision.FP32, true);
        Tensor b = Tensor.fromArray(new double[][]{{3.0, 4.0}}, Precision.FP32, true);
        
        Tensor c = TensorOps.add(a, b);
        Tensor loss = TensorOps.sum(c, 1);
        
        Tensor.backward(loss);
        
        // Gradients should be 1.0 for both inputs
        double[] aGrad = a.gradToArray();
        double[] bGrad = b.gradToArray();
        
        assertEquals(1.0, aGrad[0], 1e-5);
        assertEquals(1.0, aGrad[1], 1e-5);
        assertEquals(1.0, bGrad[0], 1e-5);
        assertEquals(1.0, bGrad[1], 1e-5);
    }

    @Test
    void testMixedPrecisionThrows() {
        Tensor a = Tensor.ones(2, 2, Precision.FP64, false);
        Tensor b = Tensor.ones(2, 2, Precision.FP32, false);
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.add(a, b)
        );
        
        assertTrue(ex.getMessage().contains("same precision"));
    }

    @Test
    void testDefaultPrecision() {
        Precision original = Tensor.getDefaultPrecision();
        
        try {
            Tensor.setDefaultPrecision(Precision.FP32);
            assertEquals(Precision.FP32, Tensor.getDefaultPrecision());
            
            Tensor t = new Tensor(2, 2, false);
            assertEquals(Precision.FP32, t.precision());
            
            Tensor.setDefaultPrecision(Precision.FP64);
            assertEquals(Precision.FP64, Tensor.getDefaultPrecision());
            
            Tensor t2 = new Tensor(2, 2, false);
            assertEquals(Precision.FP64, t2.precision());
        } finally {
            Tensor.setDefaultPrecision(original);
        }
    }

    @Test
    void testFP32RandomInitialization() {
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        
        Tensor t = Tensor.rand(10, 10, Precision.FP32, rng, -1.0, 1.0, false);
        assertEquals(Precision.FP32, t.precision());
        
        // Check values are in range
        double[] data = t.toArray();
        for (double v : data) {
            assertTrue(v >= -1.0 && v <= 1.0);
        }
    }

    @Test
    void testPrecisionMetadata() {
        assertEquals(64, Precision.FP64.bits());
        assertEquals(32, Precision.FP32.bits());
        assertEquals("float64", Precision.FP64.displayName());
        assertEquals("float32", Precision.FP32.displayName());
    }

    @Test
    void testZerosLikePreservesPrecision() {
        Tensor template = new Tensor(3, 4, Precision.FP32, false);
        Tensor zeros = Tensor.zerosLike(template, false);
        
        assertEquals(Precision.FP32, zeros.precision());
        assertEquals(3, zeros.rows());
        assertEquals(4, zeros.cols());
        assertEquals(0.0, zeros.get(0, 0), 1e-10);
    }

    @Test
    void testOnesLikePreservesPrecision() {
        Tensor template = new Tensor(2, 3, Precision.FP32, false);
        Tensor ones = Tensor.onesLike(template, false);
        
        assertEquals(Precision.FP32, ones.precision());
        assertEquals(2, ones.rows());
        assertEquals(3, ones.cols());
        assertEquals(1.0, ones.get(0, 0), 1e-6);
    }

    @Test
    void testRandLikePreservesPrecision() {
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(123);
        
        Tensor template = new Tensor(5, 5, Precision.FP32, false);
        Tensor rand = Tensor.randLike(template, rng, 0.0, 1.0, false);
        
        assertEquals(Precision.FP32, rand.precision());
        assertEquals(5, rand.rows());
        assertEquals(5, rand.cols());
    }
}

