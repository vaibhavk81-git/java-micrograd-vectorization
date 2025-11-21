package ai.micrograd;

import ai.micrograd.nn.VectorMLP;
import ai.micrograd.optim.SGD;
import ai.micrograd.tensor.Precision;
import ai.micrograd.tensor.Tensor;
import ai.micrograd.util.Profiler;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.random.RandomGenerator;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for individual components: SGD, VectorMLP, Precision, Profiler.
 */
class ComponentTests {
    
    // ========== SGD Tests ==========
    
    @Test
    void testSGDStep() {
        Tensor param = Tensor.fromArray(new double[][]{{1.0, 2.0}, {3.0, 4.0}}, true);
        
        // Set gradients
        param.grad()[0] = 0.1;
        param.grad()[1] = 0.2;
        param.grad()[2] = 0.3;
        param.grad()[3] = 0.4;
        
        SGD optimizer = new SGD(0.1);
        optimizer.step(param);
        
        // Check updates: data[i] -= lr * grad[i]
        assertEquals(1.0 - 0.1 * 0.1, param.data()[0], 1e-10);
        assertEquals(2.0 - 0.1 * 0.2, param.data()[1], 1e-10);
        assertEquals(3.0 - 0.1 * 0.3, param.data()[2], 1e-10);
        assertEquals(4.0 - 0.1 * 0.4, param.data()[3], 1e-10);
    }
    
    @Test
    void testSGDStepEquivalence() {
        Tensor param = Tensor.fromArray(new double[][]{{1.0, 2.0}}, true);
        param.grad()[0] = 0.5;
        param.grad()[1] = 1.0;
        
        double lr = 0.01;
        
        // Manual update
        double expected0 = param.data()[0] - lr * param.grad()[0];
        double expected1 = param.data()[1] - lr * param.grad()[1];
        
        // SGD update
        SGD optimizer = new SGD(lr);
        optimizer.step(param);
        
        assertEquals(expected0, param.data()[0], 1e-10);
        assertEquals(expected1, param.data()[1], 1e-10);
    }
    
    // ========== VectorMLP Tests ==========
    
    @Test
    void testVectorMLPForwardShape() {
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        
        VectorMLP model = new VectorMLP(3, new int[]{4, 5}, 2, true, rng);
        
        Tensor input = new Tensor(10, 3, false);  // Batch of 10
        Tensor output = model.forward(input);
        
        assertEquals(10, output.rows());
        assertEquals(2, output.cols());
    }
    
    @Test
    void testVectorMLPParameterCount() {
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        
        VectorMLP model = new VectorMLP(2, new int[]{4}, 1, true, rng);
        
        // Expected parameters:
        // Layer 1: W(2×4) + b(1×4) = 8 + 4 = 12
        // Layer 2: W(4×1) + b(1×1) = 4 + 1 = 5
        // Total: 17 tensors (but parameters() returns individual tensors, so 4 tensors total)
        
        assertEquals(4, model.parameters().size());  // 4 parameter tensors (2 weights + 2 biases)
        
        // Check weights only
        assertEquals(2, model.weights().size());  // 2 weight tensors
    }
    
    @Test
    void testVectorMLPZeroGrad() {
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        
        VectorMLP model = new VectorMLP(2, new int[]{3}, 1, true, rng);
        
        // Set some gradients
        for (Tensor param : model.parameters()) {
            for (int i = 0; i < param.grad().length; i++) {
                param.grad()[i] = 1.0;
            }
        }
        
        // Zero gradients
        model.zeroGrad();
        
        // Check all gradients are zero
        for (Tensor param : model.parameters()) {
            for (int i = 0; i < param.grad().length; i++) {
                assertEquals(0.0, param.grad()[i], 1e-10);
            }
        }
    }
    
    // ========== Precision Tests ==========
    
    @Test
    void testPrecisionFP64Works() {
        Tensor t = new Tensor(2, 3, Precision.FP64, false);
        assertEquals(Precision.FP64, t.precision());
        assertEquals(2, t.rows());
        assertEquals(3, t.cols());
    }
    
    @Test
    void testPrecisionFP32ThrowsException() {
        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> new Tensor(2, 3, Precision.FP32, false)
        );
        
        assertTrue(ex.getMessage().contains("FP32"));
        assertTrue(ex.getMessage().contains("not yet implemented"));
    }
    
    // ========== Profiler Tests ==========
    
    @Test
    void testProfilerTimeNanos() {
        long elapsed = Profiler.timeNanos(() -> {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        // Should take at least 10ms = 10_000_000 nanos
        assertTrue(elapsed > 0, "Elapsed time should be positive");
        assertTrue(elapsed >= 5_000_000, "Elapsed time should be at least 5ms");
    }
    
    @Test
    void testProfilerTrainStep() {
        Map<String, Long> times = Profiler.profileTrainStep(
            () -> { /* forward */ },
            () -> { /* backward */ },
            () -> { /* step */ }
        );
        
        assertTrue(times.containsKey("forward"));
        assertTrue(times.containsKey("backward"));
        assertTrue(times.containsKey("step"));
        
        assertTrue(times.get("forward") >= 0);
        assertTrue(times.get("backward") >= 0);
        assertTrue(times.get("step") >= 0);
    }
    
    @Test
    void testProfilerNanosToMillis() {
        assertEquals(1.0, Profiler.nanosToMillis(1_000_000), 1e-6);
        assertEquals(10.0, Profiler.nanosToMillis(10_000_000), 1e-6);
        assertEquals(0.001, Profiler.nanosToMillis(1_000), 1e-6);
    }
}

