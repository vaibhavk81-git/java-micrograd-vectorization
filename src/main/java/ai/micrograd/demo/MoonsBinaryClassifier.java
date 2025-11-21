package ai.micrograd.demo;

import ai.micrograd.data.Datasets;
import ai.micrograd.nn.Losses;
import ai.micrograd.nn.VectorMLP;
import ai.micrograd.optim.SGD;
import ai.micrograd.tensor.DeviceManager;
import ai.micrograd.tensor.DeviceType;
import ai.micrograd.tensor.Precision;
import ai.micrograd.tensor.Tensor;
import ai.micrograd.util.Profiler;
import ai.micrograd.visualization.DecisionBoundaryPlotter;
import ai.micrograd.visualization.LossPlotter;
import ai.micrograd.visualization.VectorMLPAdapter;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.random.RandomGenerator;

/**
 * MoonsBinaryClassifier trains a neural network on the two-moons dataset.
 * 
 * <p>This CLI application demonstrates:
 * <ul>
 *   <li>Vectorized tensor operations</li>
 *   <li>Mini-batch training with SGD</li>
 *   <li>Deterministic shuffling</li>
 *   <li>Hinge loss with L2 regularization</li>
 *   <li>Training/test split and evaluation</li>
 *   <li>Visualization of loss curves and decision boundaries</li>
 * </ul>
 * 
 * @author Vaibhav Khare
 */
@Command(name = "moons-train", mixinStandardHelpOptions = true, version = "1.0",
         description = "Trains a binary classifier on the two-moons dataset")
public class MoonsBinaryClassifier implements Callable<Integer> {
    
    @Option(names = {"--hidden"}, description = "Hidden layer width (default: ${DEFAULT-VALUE})")
    private int hidden = 32;
    
    @Option(names = {"--depth"}, description = "Number of hidden layers (default: ${DEFAULT-VALUE})")
    private int depth = 2;
    
    @Option(names = {"--epochs"}, description = "Number of epochs (default: ${DEFAULT-VALUE}, max: 200)")
    private int epochs = 150;
    
    @Option(names = {"--lr"}, description = "Learning rate (default: ${DEFAULT-VALUE})")
    private double lr = 0.04;
    
    @Option(names = {"--batch"}, description = "Batch size (default: ${DEFAULT-VALUE})")
    private int batchSize = 32;
    
    @Option(names = {"--seed"}, description = "Random seed (default: ${DEFAULT-VALUE})")
    private long seed = 42;
    
    @Option(names = {"--l2"}, description = "L2 regularization strength (default: ${DEFAULT-VALUE})")
    private double l2 = 0.025;
    
    @Option(names = {"--samples"}, description = "Number of samples (default: ${DEFAULT-VALUE})")
    private int samples = 1000;
    
    @Option(names = {"--noise"}, description = "Dataset noise level (default: ${DEFAULT-VALUE})")
    private double noise = 0.12;
    
    @Option(names = {"--outDir"}, description = "Output directory (default: runs/<timestamp>/)")
    private String outDir = null;
    
    @Option(names = {"--precision"}, description = "Precision mode: FP64 or FP32 (default: ${DEFAULT-VALUE})")
    private Precision precision = Precision.FP64;
    
    @Option(names = {"--device"}, description = "Device type (cpu). GPU will be enabled once available.")
    private String deviceFlag = DeviceManager.get().defaultDevice().cliValue();
    
    @Option(names = {"--profile"}, description = "Enable profiling (default: ${DEFAULT-VALUE})")
    private boolean profile = false;
    
    public static void main(String[] args) {
        int exitCode = new CommandLine(new MoonsBinaryClassifier()).execute(args);
        System.exit(exitCode);
    }
    
    @Override
    public Integer call() throws Exception {
        // Validate inputs
        if (epochs > 200) {
            System.err.println("Error: epochs must be <= 200, got: " + epochs);
            return 1;
        }
        if (hidden < 1) {
            System.err.println("Error: hidden must be >= 1, got: " + hidden);
            return 1;
        }
        if (depth < 1) {
            System.err.println("Error: depth must be >= 1, got: " + depth);
            return 1;
        }
        
        DeviceManager deviceManager = DeviceManager.get();
        DeviceType selectedDevice;
        try {
            selectedDevice = DeviceType.fromString(deviceFlag);
        } catch (IllegalArgumentException ex) {
            System.err.println("Error: unknown device '" + deviceFlag + "'. Valid options: " +
                formatDevices(deviceManager.availableDevices()));
            return 1;
        }
        if (!deviceManager.isAvailable(selectedDevice)) {
            System.err.println("Error: device '" + deviceFlag + "' is not available. Supported: " +
                formatDevices(deviceManager.availableDevices()));
            deviceManager.gpuRequestWarning().ifPresent(System.err::println);
            return 1;
        }
        deviceManager.gpuRequestWarning().ifPresent(System.out::println);
        Tensor.setDefaultPrecision(precision);
        
        // Setup output directory
        if (outDir == null) {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
            outDir = "runs/" + timestamp;
        }
        Path outPath = Path.of(outDir);
        Files.createDirectories(outPath);
        System.out.println("Output directory: " + outPath.toAbsolutePath());
        
        // Initialize RNGs
        RandomGenerator initRng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(seed);
        RandomGenerator dataRng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(seed ^ 0x9E3779B97F4A7C15L);
        
        System.out.println("\n=== Configuration ===");
        System.out.println("Hidden size: " + hidden);
        System.out.println("Epochs: " + epochs);
        System.out.println("Learning rate: " + lr);
        System.out.println("Batch size: " + batchSize);
        System.out.println("Seed: " + seed);
        System.out.println("L2: " + l2);
        System.out.println("Samples: " + samples);
        System.out.println("Noise: " + noise);
        System.out.println("Precision: " + precision);
        System.out.println("Device: " + selectedDevice.cliValue());
        System.out.println("Profile: " + profile);
        
        // Generate dataset
        System.out.println("\n=== Generating Dataset ===");
        Datasets.MoonsData data = Datasets.makeMoons(samples, noise, seed);
        Tensor X_data = data.X().to(selectedDevice, precision);
        Tensor y_data = data.y().to(selectedDevice, precision);
        System.out.println("X shape: (" + X_data.rows() + ", " + X_data.cols() + ")");
        System.out.println("y shape: (" + y_data.rows() + ", " + y_data.cols() + ")");
        
        // Train/test split (80/20)
        int trainSize = (int) (samples * 0.8);
        int testSize = samples - trainSize;
        
        Tensor X_train = extractRows(X_data, 0, trainSize);
        Tensor y_train = extractRows(y_data, 0, trainSize);
        Tensor X_test = extractRows(X_data, trainSize, samples);
        Tensor y_test = extractRows(y_data, trainSize, samples);
        
        System.out.println("Train: " + trainSize + " samples");
        System.out.println("Test: " + testSize + " samples");
        
        // Create model
        System.out.println("\n=== Model ===");
        int[] hiddenDims = new int[depth];
        Arrays.fill(hiddenDims, hidden);
        VectorMLP model = new VectorMLP(2, hiddenDims, 1, true, initRng);
        System.out.println(model);
        System.out.println("Parameters: " + model.parameters().size());
        
        // Create optimizer
        SGD optimizer = new SGD(lr);
        System.out.println("Optimizer: " + optimizer);
        
        // Training loop
        System.out.println("\n=== Training ===");
        List<Double> lossHistory = new ArrayList<>();
        List<Double> trainAccHistory = new ArrayList<>();
        List<Double> testAccHistory = new ArrayList<>();
        Map<String, Long> totalTimes = new LinkedHashMap<>();
        if (profile) {
            totalTimes.put("forward", 0L);
            totalTimes.put("backward", 0L);
            totalTimes.put("step", 0L);
        }
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data
            int[] indices = shuffle(trainSize, dataRng);
            
            double epochLoss = 0.0;
            int numBatches = 0;
            
            // Mini-batch training
            for (int start = 0; start < trainSize; start += batchSize) {
                int end = Math.min(start + batchSize, trainSize);
                int[] batchIndices = Arrays.copyOfRange(indices, start, end);
                
                Tensor X_batch = extractRowsByIndices(X_train, batchIndices);
                Tensor y_batch = extractRowsByIndices(y_train, batchIndices);
                
                // Forward, backward, step
                if (profile) {
                    Tensor[] scoreHolder = new Tensor[1];
                    Tensor[] lossHolder = new Tensor[1];
                    
                    Map<String, Long> times = Profiler.profileTrainStep(
                        () -> {
                            scoreHolder[0] = model.forward(X_batch);
                            lossHolder[0] = Losses.hingeLossWithL2(scoreHolder[0], y_batch, model.weights(), l2);
                        },
                        () -> Tensor.backward(lossHolder[0]),
                        () -> optimizer.step(model.parameters())
                    );
                    
                    for (Map.Entry<String, Long> entry : times.entrySet()) {
                        totalTimes.put(entry.getKey(), totalTimes.get(entry.getKey()) + entry.getValue());
                    }
                    
                    epochLoss += lossHolder[0].item();
                } else {
                    Tensor score = model.forward(X_batch);
                    Tensor loss = Losses.hingeLossWithL2(score, y_batch, model.weights(), l2);
                    
                    Tensor.backward(loss);
                    optimizer.step(model.parameters());
                    
                    epochLoss += loss.item();
                }
                
                model.zeroGrad();
                numBatches++;
            }
            
            epochLoss /= numBatches;
            lossHistory.add(epochLoss);
            
            // Evaluate
            double trainAcc = evaluate(model, X_train, y_train);
            double testAcc = evaluate(model, X_test, y_test);
            trainAccHistory.add(trainAcc);
            testAccHistory.add(testAcc);
            
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                System.out.printf("Epoch %3d/%d | Loss: %.4f | Train Acc: %.3f | Test Acc: %.3f%n",
                    epoch + 1, epochs, epochLoss, trainAcc, testAcc);
            }
        }
        
        // Final evaluation
        double finalTrainAcc = evaluate(model, X_train, y_train);
        double finalTestAcc = evaluate(model, X_test, y_test);
        System.out.println("\n=== Final Results ===");
        System.out.printf("Train Accuracy: %.4f%n", finalTrainAcc);
        System.out.printf("Test Accuracy: %.4f%n", finalTestAcc);
        
        if (profile) {
            System.out.println("\n=== Profiling Results ===");
            System.out.println("Total times:");
            for (Map.Entry<String, Long> entry : totalTimes.entrySet()) {
                System.out.printf("  %s: %.2f ms%n", entry.getKey(), Profiler.nanosToMillis(entry.getValue()));
            }
        }
        
        // Generate plots
        System.out.println("\n=== Generating Plots ===");
        LossPlotter.plot(lossHistory, outPath.resolve("loss.png"));
        DecisionBoundaryPlotter.plot(new VectorMLPAdapter(model), data.X(), data.y(), outPath.resolve("decision_boundary.png"));
        System.out.println("Plots saved to " + outPath.toAbsolutePath());
        
        // Save metrics
        saveMetrics(outPath.resolve("metrics.json"), finalTrainAcc, finalTestAcc, 
                    lossHistory.get(lossHistory.size() - 1), totalTimes);
        
        System.out.println("\nTraining complete!");
        return 0;
    }
    
    private Tensor extractRows(Tensor t, int start, int end) {
        int rows = end - start;
        Tensor result = new Tensor(rows, t.cols(), false);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < t.cols(); c++) {
                result.set(r, c, t.get(start + r, c));
            }
        }
        return result;
    }
    
    private Tensor extractRowsByIndices(Tensor t, int[] indices) {
        Tensor result = new Tensor(indices.length, t.cols(), false);
        for (int i = 0; i < indices.length; i++) {
            for (int c = 0; c < t.cols(); c++) {
                result.set(i, c, t.get(indices[i], c));
            }
        }
        return result;
    }
    
    private int[] shuffle(int n, RandomGenerator rng) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }
    
    private double evaluate(VectorMLP model, Tensor X, Tensor y) {
        Tensor score = model.forward(X);
        int correct = 0;
        for (int i = 0; i < X.rows(); i++) {
            double pred = score.get(i, 0) >= 0 ? 1.0 : -1.0;
            if (pred == y.get(i, 0)) {
                correct++;
            }
        }
        return (double) correct / X.rows();
    }
    
    private void saveMetrics(Path file, double trainAcc, double testAcc, double finalLoss, 
                             Map<String, Long> times) throws IOException {
        try (FileWriter writer = new FileWriter(file.toFile())) {
            writer.write("{\n");
            writer.write("  \"hidden\": " + hidden + ",\n");
            writer.write("  \"epochs\": " + epochs + ",\n");
            writer.write("  \"lr\": " + lr + ",\n");
            writer.write("  \"batch_size\": " + batchSize + ",\n");
            writer.write("  \"seed\": " + seed + ",\n");
            writer.write("  \"l2\": " + l2 + ",\n");
            writer.write("  \"samples\": " + samples + ",\n");
            writer.write("  \"noise\": " + noise + ",\n");
            writer.write("  \"train_accuracy\": " + trainAcc + ",\n");
            writer.write("  \"test_accuracy\": " + testAcc + ",\n");
            writer.write("  \"final_loss\": " + finalLoss);
            
            if (profile && times != null && !times.isEmpty()) {
                writer.write(",\n  \"profiling\": {\n");
                int i = 0;
                for (Map.Entry<String, Long> entry : times.entrySet()) {
                    writer.write("    \"" + entry.getKey() + "_ms\": " + 
                                 Profiler.nanosToMillis(entry.getValue()));
                    if (i < times.size() - 1) writer.write(",");
                    writer.write("\n");
                    i++;
                }
                writer.write("  }");
            }
            
            writer.write("\n}\n");
        }
        System.out.println("Saved: " + file.getFileName());
    }

    private String formatDevices(List<DeviceType> devices) {
        List<String> values = new ArrayList<>();
        for (DeviceType device : devices) {
            values.add(device.cliValue());
        }
        return String.join(", ", values);
    }
}

