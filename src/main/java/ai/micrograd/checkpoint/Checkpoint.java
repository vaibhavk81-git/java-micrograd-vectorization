package ai.micrograd.checkpoint;

import ai.micrograd.tensor.Tensor;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Checkpoint for saving and loading MLP training state.
 * 
 * <p>Stores complete training state including:
 * <ul>
 *   <li>Model weights (as serialized arrays)</li>
 *   <li>Hyperparameters (architecture, learning rate, etc.)</li>
 *   <li>Training metrics (epoch, losses, accuracies)</li>
 *   <li>Timestamp</li>
 * </ul>
 * 
 * <p>This enables:
 * <ul>
 *   <li>Resuming training from interruption</li>
 *   <li>Loading best model for inference</li>
 *   <li>Comparing different training runs</li>
 *   <li>Reproducing results</li>
 * </ul>
 * 
 * <p>Example usage:
 * <pre>
 * Checkpoint checkpoint = new Checkpoint.Builder()
 *     .setModelWeights(model.parameters())
 *     .setHyperparameters(hyperparams)
 *     .setEpoch(epoch)
 *     .setTrainLoss(trainLoss)
 *     .setTestLoss(testLoss)
 *     .setTrainAccuracy(trainAcc)
 *     .setTestAccuracy(testAcc)
 *     .build();
 * 
 * checkpoint.save(path);
 * 
 * Checkpoint loaded = Checkpoint.load(path);
 * </pre>
 */
public class Checkpoint {
    private static final int CURRENT_VERSION = 1;
    
    private Map<String, double[][]> modelWeights;
    private Map<String, Object> hyperparameters;
    private int epoch;
    private double trainLoss;
    private double testLoss;
    private double trainAccuracy;
    private double testAccuracy;
    private long timestamp;
    private int version = CURRENT_VERSION;
    
    private Checkpoint() {
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Saves checkpoint to a file.
     * 
     * @param path output file path
     * @throws IOException if file cannot be written
     */
    public void save(Path path) throws IOException {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        
        // Convert to serializable format
        CheckpointData data = new CheckpointData();
        data.modelWeights = this.modelWeights;
        data.hyperparameters = this.hyperparameters;
        data.epoch = this.epoch;
        data.trainLoss = this.trainLoss;
        data.testLoss = this.testLoss;
        data.trainAccuracy = this.trainAccuracy;
        data.testAccuracy = this.testAccuracy;
        data.timestamp = this.timestamp;
        data.version = CURRENT_VERSION;
        
        Files.createDirectories(path.getParent());
        try (Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            gson.toJson(data, writer);
        }
        
        System.out.println("Saved checkpoint to " + path + " (epoch " + epoch + 
                          ", test_loss=" + String.format("%.6f", testLoss) + ")");
    }
    
    /**
     * Loads checkpoint from a file.
     * 
     * @param path checkpoint file path
     * @return loaded checkpoint
     * @throws IOException if file cannot be read
     */
    public static Checkpoint load(Path path) throws IOException {
        Gson gson = new Gson();
        
        CheckpointData data;
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            data = gson.fromJson(reader, CheckpointData.class);
        }
        
        Checkpoint checkpoint = new Checkpoint();
        checkpoint.modelWeights = data.modelWeights;
        checkpoint.hyperparameters = data.hyperparameters;
        checkpoint.epoch = data.epoch;
        checkpoint.trainLoss = data.trainLoss;
        checkpoint.testLoss = data.testLoss;
        checkpoint.trainAccuracy = data.trainAccuracy;
        checkpoint.testAccuracy = data.testAccuracy;
        checkpoint.timestamp = data.timestamp;
        checkpoint.version = data.version == 0 ? 1 : data.version;
        
        if (checkpoint.version > CURRENT_VERSION) {
            throw new IllegalStateException(String.format(
                "Checkpoint version %d is newer than runtime (%d). Please upgrade.",
                checkpoint.version, CURRENT_VERSION));
        }
        
        System.out.println("Loaded checkpoint from " + path + " (epoch " + checkpoint.epoch + 
                          ", test_loss=" + String.format("%.6f", checkpoint.testLoss) + ")");
        return checkpoint;
    }
    
    public Map<String, double[][]> getModelWeights() {
        return modelWeights;
    }
    
    public Map<String, Object> getHyperparameters() {
        return hyperparameters;
    }
    
    public int getEpoch() {
        return epoch;
    }
    
    public double getTrainLoss() {
        return trainLoss;
    }
    
    public double getTestLoss() {
        return testLoss;
    }
    
    public double getTrainAccuracy() {
        return trainAccuracy;
    }
    
    public double getTestAccuracy() {
        return testAccuracy;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * Loads weights from checkpoint into a model.
     * 
     * @param model model to load weights into
     * @throws IllegalArgumentException if checkpoint weights don't match model architecture
     */
    public void loadIntoModel(ai.micrograd.nn.VectorMLP model) {
        List<Tensor> parameters = model.parameters();
        
        if (parameters.size() != modelWeights.size()) {
            throw new IllegalArgumentException(
                String.format("Checkpoint has %d parameters, but model has %d",
                    modelWeights.size(), parameters.size()));
        }
        
        for (int i = 0; i < parameters.size(); i++) {
            String key = "param_" + i;
            double[][] weightData = modelWeights.get(key);
            if (weightData == null) {
                throw new IllegalArgumentException("Missing weight: " + key);
            }
            
            Tensor param = parameters.get(i);
            if (param.rows() != weightData.length || param.cols() != weightData[0].length) {
                throw new IllegalArgumentException(
                    String.format("Weight %s shape mismatch: checkpoint (%d, %d) vs model (%d, %d)",
                        key, weightData.length, weightData[0].length, param.rows(), param.cols()));
            }
            
            // Copy weights into tensor
            for (int r = 0; r < param.rows(); r++) {
                for (int c = 0; c < param.cols(); c++) {
                    param.set(r, c, weightData[r][c]);
                }
            }
        }
    }
    
    /**
     * Builder for creating checkpoints.
     */
    public static class Builder {
        private final Checkpoint checkpoint;
        
        public Builder() {
            this.checkpoint = new Checkpoint();
            this.checkpoint.modelWeights = new HashMap<>();
            this.checkpoint.hyperparameters = new HashMap<>();
        }
        
        public Builder setModelWeights(List<Tensor> parameters) {
            for (int i = 0; i < parameters.size(); i++) {
                Tensor param = parameters.get(i);
                double[] flat = param.toArray();
                double[][] array2D = new double[param.rows()][param.cols()];
                for (int r = 0; r < param.rows(); r++) {
                    System.arraycopy(flat, r * param.cols(), array2D[r], 0, param.cols());
                }
                checkpoint.modelWeights.put("param_" + i, array2D);
            }
            return this;
        }
        
        public Builder setHyperparameters(Map<String, Object> hyperparameters) {
            checkpoint.hyperparameters = new HashMap<>(hyperparameters);
            return this;
        }
        
        public Builder setHyperparameter(String key, Object value) {
            checkpoint.hyperparameters.put(key, value);
            return this;
        }
        
        public Builder setEpoch(int epoch) {
            checkpoint.epoch = epoch;
            return this;
        }
        
        public Builder setTrainLoss(double trainLoss) {
            checkpoint.trainLoss = trainLoss;
            return this;
        }
        
        public Builder setTestLoss(double testLoss) {
            checkpoint.testLoss = testLoss;
            return this;
        }
        
        public Builder setTrainAccuracy(double trainAccuracy) {
            checkpoint.trainAccuracy = trainAccuracy;
            return this;
        }
        
        public Builder setTestAccuracy(double testAccuracy) {
            checkpoint.testAccuracy = testAccuracy;
            return this;
        }
        
        public Checkpoint build() {
            return checkpoint;
        }
    }
    
    /**
     * Internal class for JSON serialization.
     */
    private static class CheckpointData {
        Map<String, double[][]> modelWeights;
        Map<String, Object> hyperparameters;
        int epoch;
        double trainLoss;
        double testLoss;
        double trainAccuracy;
        double testAccuracy;
        long timestamp;
        int version;
    }
}

