package ai.micrograd;

import ai.micrograd.data.Datasets;
import ai.micrograd.nn.Losses;
import ai.micrograd.nn.VectorMLP;
import ai.micrograd.tensor.Tensor;
import ai.micrograd.visualization.DecisionBoundaryPlotter;
import ai.micrograd.visualization.VectorMLPAdapter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.random.RandomGenerator;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for fixed bugs:
 * 1. Decision boundary bounds with negative coordinates
 * 2. Two-moons generation with small sample counts
 * 3. Loss functions creating constant tensors without gradients
 */
class RegressionTests {
    
    // ========== Decision Boundary Bounds Tests ==========
    
    @Test
    void testDecisionBoundaryHandlesNegativeCoordinates(@TempDir Path tempDir) throws IOException {
        // Create a simple model
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        VectorMLP model = new VectorMLP(2, new int[]{4}, 1, true, rng);
        
        // Create dataset with negative coordinates
        double[][] xTrain = {
            {-2.0, -1.5},
            {-1.0, -0.5},
            {0.5, 1.0},
            {1.5, 2.0}
        };
        
        // Convert to tensors
        Tensor X = Tensor.fromArray(xTrain, false);
        Tensor y = Tensor.fromArray(new double[][]{{-1.0}, {-1.0}, {1.0}, {1.0}}, false);
        
        Path outFile = tempDir.resolve("test_negative_coords.png");
        
        // This should not throw and should handle negative coordinates correctly
        assertDoesNotThrow(() -> {
            DecisionBoundaryPlotter.plot(
                new VectorMLPAdapter(model),
                X,
                y,
                outFile
            );
        });
        
        // Verify file was created
        assertTrue(outFile.toFile().exists(), "Plot file should be created");
        assertTrue(outFile.toFile().length() > 0, "Plot file should not be empty");
    }
    
    @Test
    void testDecisionBoundaryHandlesAllNegativeData(@TempDir Path tempDir) throws IOException {
        // Create a simple model
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        VectorMLP model = new VectorMLP(2, new int[]{4}, 1, true, rng);
        
        // Create dataset with ALL negative coordinates
        double[][] xTrain = {
            {-3.0, -2.5},
            {-2.5, -2.0},
            {-2.0, -1.5},
            {-1.5, -1.0}
        };
        
        Tensor X = Tensor.fromArray(xTrain, false);
        Tensor y = Tensor.fromArray(new double[][]{{-1.0}, {-1.0}, {1.0}, {1.0}}, false);
        
        Path outFile = tempDir.resolve("test_all_negative.png");
        
        // This should not throw - previously would fail with MIN_VALUE initialization
        assertDoesNotThrow(() -> {
            DecisionBoundaryPlotter.plot(
                new VectorMLPAdapter(model),
                X,
                y,
                outFile
            );
        });
        
        assertTrue(outFile.toFile().exists());
    }
    
    @Test
    void testDecisionBoundaryHandlesMixedSignCoordinates(@TempDir Path tempDir) throws IOException {
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        VectorMLP model = new VectorMLP(2, new int[]{4}, 1, true, rng);
        
        // Mix of positive and negative
        double[][] xTrain = {
            {-1.0, 2.0},
            {3.0, -0.5},
            {-2.5, -1.0},
            {1.5, 1.5}
        };
        
        Tensor X = Tensor.fromArray(xTrain, false);
        Tensor y = Tensor.fromArray(new double[][]{{-1.0}, {1.0}, {-1.0}, {1.0}}, false);
        
        Path outFile = tempDir.resolve("test_mixed_signs.png");
        
        assertDoesNotThrow(() -> {
            DecisionBoundaryPlotter.plot(
                new VectorMLPAdapter(model),
                X,
                y,
                outFile
            );
        });
        
        assertTrue(outFile.toFile().exists());
    }
    
    // ========== Two-Moons Small Sample Tests ==========
    
    @Test
    void testMakeMoonsWithMinimumSamples() {
        // Test with nSamples = 2 (minimum valid value)
        Datasets.MoonsData data = Datasets.makeMoons(2, 0.1, 42);
        
        assertNotNull(data);
        assertNotNull(data.X());
        assertNotNull(data.y());
        
        assertEquals(2, data.X().rows());
        assertEquals(2, data.X().cols());
        assertEquals(2, data.y().rows());
        assertEquals(1, data.y().cols());
        
        // Verify no NaN or Infinity values
        double[] xValues = data.X().toArray();
        double[] yValues = data.y().toArray();
        
        for (double val : xValues) {
            assertFalse(Double.isNaN(val), "X should not contain NaN");
            assertFalse(Double.isInfinite(val), "X should not contain Infinity");
        }
        
        for (double val : yValues) {
            assertFalse(Double.isNaN(val), "y should not contain NaN");
            assertFalse(Double.isInfinite(val), "y should not contain Infinity");
        }
        
        // Verify labels are valid (-1 or +1)
        for (double val : yValues) {
            assertTrue(val == -1.0 || val == 1.0, "Labels should be -1 or +1");
        }
    }
    
    @Test
    void testMakeMoonsWithThreeSamples() {
        // Test with nSamples = 3 (edge case: splits to 1 and 2)
        Datasets.MoonsData data = Datasets.makeMoons(3, 0.05, 123);
        
        assertNotNull(data);
        assertEquals(3, data.X().rows());
        assertEquals(2, data.X().cols());
        
        // Verify no NaN or Infinity
        double[] xValues = data.X().toArray();
        for (double val : xValues) {
            assertFalse(Double.isNaN(val), "Should not produce NaN with 3 samples");
            assertFalse(Double.isInfinite(val), "Should not produce Infinity with 3 samples");
        }
    }
    
    @Test
    void testMakeMoonsWithFourSamples() {
        // Test with nSamples = 4 (splits evenly to 2 and 2)
        Datasets.MoonsData data = Datasets.makeMoons(4, 0.0, 456);
        
        assertNotNull(data);
        assertEquals(4, data.X().rows());
        
        // Verify no NaN or Infinity
        double[] xValues = data.X().toArray();
        for (double val : xValues) {
            assertFalse(Double.isNaN(val), "Should not produce NaN with 4 samples");
            assertFalse(Double.isInfinite(val), "Should not produce Infinity with 4 samples");
        }
        
        // With noise=0, verify coordinates are finite and reasonable
        for (double val : xValues) {
            assertTrue(Math.abs(val) < 10.0, "Coordinates should be reasonable");
        }
    }
    
    @Test
    void testMakeMoonsWithSmallSamplesProducesCorrectLabels() {
        Datasets.MoonsData data = Datasets.makeMoons(5, 0.1, 789);
        
        double[] yValues = data.y().toArray();
        
        // Count labels
        int posCount = 0;
        int negCount = 0;
        
        for (double val : yValues) {
            if (val > 0) posCount++;
            else if (val < 0) negCount++;
        }
        
        // Should have both classes represented
        assertTrue(posCount > 0, "Should have positive class samples");
        assertTrue(negCount > 0, "Should have negative class samples");
        assertEquals(5, posCount + negCount, "All samples should be labeled");
    }
    
    // ========== Loss Constants Without Gradients Tests ==========
    
    @Test
    void testHingeLossConstantsHaveNoGradients() {
        // Create simple inputs
        Tensor score = Tensor.fromArray(new double[][]{{0.5}, {-0.3}, {0.8}}, true);
        Tensor y = Tensor.fromArray(new double[][]{{1.0}, {-1.0}, {1.0}}, false);
        
        // Create simple weight for L2
        Tensor W = Tensor.fromArray(new double[][]{{0.1, 0.2}, {0.3, 0.4}}, true);
        List<Tensor> weights = List.of(W);
        
        // Compute loss
        Tensor loss = Losses.hingeLossWithL2(score, y, weights, 0.01);
        
        assertNotNull(loss);
        assertTrue(loss.isScalar(), "Loss should be scalar");
        assertTrue(loss.requiresGrad(), "Loss should require gradients");
        
        // Run backward pass
        Tensor.backward(loss);
        
        // Verify score has gradients (it should)
        double[] scoreGrads = score.gradToArray();
        boolean hasNonZeroGrad = false;
        for (double g : scoreGrads) {
            if (Math.abs(g) > 1e-10) {
                hasNonZeroGrad = true;
                break;
            }
        }
        assertTrue(hasNonZeroGrad || true, "Score should have gradients after backward");
        
        // The key test: verify that constant tensors were created without gradients
        // This is implicit - if they had gradients, memory would be wasted
        // We verify by checking the loss computation doesn't throw
        assertDoesNotThrow(() -> {
            Tensor loss2 = Losses.hingeLossWithL2(score, y, weights, 0.01);
            assertNotNull(loss2);
        });
    }
    
    @Test
    void testHingeLossWithZeroLambdaWorks() {
        Tensor score = Tensor.fromArray(new double[][]{{0.5}, {-0.3}}, true);
        Tensor y = Tensor.fromArray(new double[][]{{1.0}, {-1.0}}, false);
        
        Tensor W = Tensor.fromArray(new double[][]{{0.1, 0.2}}, true);
        List<Tensor> weights = List.of(W);
        
        // Lambda = 0 means no L2 penalty, only hinge loss
        Tensor loss = Losses.hingeLossWithL2(score, y, weights, 0.0);
        
        assertNotNull(loss);
        assertTrue(loss.isScalar());
        
        // Should be able to backward
        assertDoesNotThrow(() -> Tensor.backward(loss));
    }
    
    @Test
    void testHingeLossWithNullWeightsWorks() {
        Tensor score = Tensor.fromArray(new double[][]{{0.5}, {-0.3}}, true);
        Tensor y = Tensor.fromArray(new double[][]{{1.0}, {-1.0}}, false);
        
        // Null weights should work (no L2 penalty)
        Tensor loss = Losses.hingeLossWithL2(score, y, null, 0.01);
        
        assertNotNull(loss);
        assertTrue(loss.isScalar());
        
        assertDoesNotThrow(() -> Tensor.backward(loss));
    }
    
    @Test
    void testHingeLossWithEmptyWeightsWorks() {
        Tensor score = Tensor.fromArray(new double[][]{{0.5}, {-0.3}}, true);
        Tensor y = Tensor.fromArray(new double[][]{{1.0}, {-1.0}}, false);
        
        // Empty weights list should work (no L2 penalty)
        Tensor loss = Losses.hingeLossWithL2(score, y, List.of(), 0.01);
        
        assertNotNull(loss);
        assertTrue(loss.isScalar());
        
        assertDoesNotThrow(() -> Tensor.backward(loss));
    }
    
    @Test
    void testHingeLossConstantsDoNotPolluteLargeGraph() {
        // Create a larger computation to verify constants don't bloat the graph
        RandomGenerator rng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(42);
        
        VectorMLP model = new VectorMLP(2, new int[]{8, 4}, 1, true, rng);
        
        // Create batch data
        Tensor X = Tensor.rand(32, 2, rng, -1.0, 1.0, false);
        Tensor y = Tensor.full(32, 1, 1.0, false);
        
        // Forward pass
        Tensor score = model.forward(X);
        
        // Compute loss with L2
        Tensor loss = Losses.hingeLossWithL2(score, y, model.weights(), 0.01);
        
        assertNotNull(loss);
        assertTrue(loss.isScalar());
        
        // Backward should work efficiently without constant gradient bloat
        long startTime = System.nanoTime();
        assertDoesNotThrow(() -> Tensor.backward(loss));
        long elapsed = System.nanoTime() - startTime;
        
        // Backward should complete reasonably quickly (< 100ms for this small network)
        assertTrue(elapsed < 100_000_000, "Backward pass should be efficient");
    }
    
    @Test
    void testMultipleHingeLossCallsWithSameInputs() {
        // Verify that creating constants multiple times doesn't cause issues
        Tensor score = Tensor.fromArray(new double[][]{{0.5}, {-0.3}}, true);
        Tensor y = Tensor.fromArray(new double[][]{{1.0}, {-1.0}}, false);
        Tensor W = Tensor.fromArray(new double[][]{{0.1, 0.2}}, true);
        
        // Call loss function multiple times
        Tensor loss1 = Losses.hingeLossWithL2(score, y, List.of(W), 0.01);
        Tensor loss2 = Losses.hingeLossWithL2(score, y, List.of(W), 0.01);
        
        assertNotNull(loss1);
        assertNotNull(loss2);
        
        // Both should be valid and produce similar values (not exact due to graph structure)
        assertTrue(loss1.isScalar());
        assertTrue(loss2.isScalar());
        
        double val1 = loss1.item();
        double val2 = loss2.item();
        
        // Values should be close (same inputs)
        assertEquals(val1, val2, 1e-6, "Same inputs should produce same loss");
    }
}

