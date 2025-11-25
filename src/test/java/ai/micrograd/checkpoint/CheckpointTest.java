package ai.micrograd.checkpoint;

import ai.micrograd.nn.VectorMLP;
import ai.micrograd.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CheckpointTest {

    @Test
    void checkpointRoundTripPreservesModelWeights(@TempDir Path tempDir) throws IOException {
        RandomGenerator rng = RandomGeneratorFactory.of("L64X128MixRandom").create(42);
        VectorMLP original = new VectorMLP(2, new int[]{4, 3}, 1, true, rng);

        // Create checkpoint from original model
        Checkpoint checkpoint = new Checkpoint.Builder()
            .setModelWeights(original.parameters())
            .setHyperparameter("hidden0", 4)
            .setHyperparameter("hidden1", 3)
            .setEpoch(5)
            .setTrainLoss(0.27)
            .setTestLoss(0.31)
            .setTrainAccuracy(0.89)
            .setTestAccuracy(0.86)
            .build();

        Path file = tempDir.resolve("checkpoint.json");
        checkpoint.save(file);

        // Create a fresh model with different initialization
        VectorMLP loaded = new VectorMLP(2, new int[]{4, 3}, 1, true,
            RandomGeneratorFactory.of("L64X128MixRandom").create(999));

        // Load weights from checkpoint
        Checkpoint loadedCkpt = Checkpoint.load(file);
        loadedCkpt.loadIntoModel(loaded);

        // Ensure predictions match
        Tensor input = Tensor.fromArray(new double[][]{{1.4, -0.5}, {0.2, 0.8}}, false);
        Tensor reference = original.forward(input);
        Tensor restored = loaded.forward(input);

        assertArrayEquals(reference.toArray(), restored.toArray(), 1e-9,
            "Model outputs should match after loading checkpoint");
        assertEquals(5, loadedCkpt.getEpoch());
        assertEquals(0.89, loadedCkpt.getTrainAccuracy(), 1e-12);
    }
}
