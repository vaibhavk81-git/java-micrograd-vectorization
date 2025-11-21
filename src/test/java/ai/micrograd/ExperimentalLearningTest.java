package ai.micrograd;

import ai.micrograd.data.Datasets;
import ai.micrograd.nn.Losses;
import ai.micrograd.nn.VectorMLP;
import ai.micrograd.optim.SGD;
import ai.micrograd.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Experimental Learning Test Suite
 * 
 * <p>This test runs a series of experiments to learn different neural network concepts:
 * - Overfitting and regularization
 * - Learning rate effects
 * - Model capacity
 * - Dataset effects
 * - Batch size effects
 * 
 * <p>Results are collected and written to a comprehensive report.
 */
class ExperimentalLearningTest {
    
    /**
     * Configuration for a single experiment
     */
    static class ExperimentConfig {
        final String name;
        final String phase;
        final String concept;
        final int hidden;
        final int epochs;
        final double lr;
        final int batchSize;
        final double l2;
        final int samples;
        final double noise;
        final long seed;
        
        ExperimentConfig(String name, String phase, String concept,
                        int hidden, int epochs, double lr, int batchSize,
                        double l2, int samples, double noise, long seed) {
            this.name = name;
            this.phase = phase;
            this.concept = concept;
            this.hidden = hidden;
            this.epochs = epochs;
            this.lr = lr;
            this.batchSize = batchSize;
            this.l2 = l2;
            this.samples = samples;
            this.noise = noise;
            this.seed = seed;
        }
    }
    
    /**
     * Results from a single experiment
     */
    static class ExperimentResult {
        final ExperimentConfig config;
        final double trainAccuracy;
        final double testAccuracy;
        final double finalLoss;
        final double accuracyGap;
        final List<Double> trainAccHistory;
        final List<Double> testAccHistory;
        final List<Double> lossHistory;
        final long durationMs;
        
        ExperimentResult(ExperimentConfig config, double trainAccuracy, double testAccuracy,
                        double finalLoss, List<Double> trainAccHistory, List<Double> testAccHistory,
                        List<Double> lossHistory, long durationMs) {
            this.config = config;
            this.trainAccuracy = trainAccuracy;
            this.testAccuracy = testAccuracy;
            this.finalLoss = finalLoss;
            this.accuracyGap = trainAccuracy - testAccuracy;
            this.trainAccHistory = trainAccHistory;
            this.testAccHistory = testAccHistory;
            this.lossHistory = lossHistory;
            this.durationMs = durationMs;
        }
    }
    
    /**
     * Define all experiments from the learning plan
     */
    private List<ExperimentConfig> defineExperiments() {
        List<ExperimentConfig> experiments = new ArrayList<>();
        long baseSeed = 42;
        
        // Underfitting scenarios
        experiments.add(new ExperimentConfig(
            "U1", "Concept: Underfitting", "Tiny network, too few parameters",
            2, 25, 0.10, 32, 0.01, 400, 0.20, baseSeed));
        experiments.add(new ExperimentConfig(
            "U2", "Concept: Underfitting", "Strong regularization suppresses learning",
            4, 120, 0.08, 32, 0.20, 400, 0.20, baseSeed));
        
        // Overfitting scenarios
        experiments.add(new ExperimentConfig(
            "O1", "Concept: Overfitting", "No regularization + many epochs",
            32, 200, 0.10, 32, 0.00, 400, 0.10, baseSeed));
        experiments.add(new ExperimentConfig(
            "O2", "Concept: Overfitting", "Tiny batches + high learning rate",
            12, 160, 0.20, 8, 0.00, 400, 0.20, baseSeed));
        
        // Learning-rate extremes
        experiments.add(new ExperimentConfig(
            "LR_LOW", "Concept: Learning Rate", "Very low lr learns slowly but steadily",
            8, 130, 0.01, 32, 0.02, 400, 0.20, baseSeed));
        experiments.add(new ExperimentConfig(
            "LR_HIGH", "Concept: Learning Rate", "Very high lr destabilizes training",
            8, 60, 0.40, 32, 0.02, 400, 0.20, baseSeed));
        
        // Regularization comparison
        experiments.add(new ExperimentConfig(
            "REG_NONE", "Concept: Regularization", "Regularization disabled",
            16, 120, 0.05, 32, 0.00, 400, 0.20, baseSeed));
        experiments.add(new ExperimentConfig(
            "REG_GOOD", "Concept: Regularization", "Balanced L2 for generalization",
            16, 120, 0.05, 32, 0.05, 400, 0.20, baseSeed));
        
        // Dataset + batch impacts
        experiments.add(new ExperimentConfig(
            "DATA_NOISY", "Concept: Dataset", "High-noise dataset",
            10, 120, 0.07, 32, 0.03, 400, 0.45, baseSeed));
        experiments.add(new ExperimentConfig(
            "DATA_CLEAN", "Concept: Dataset", "Clean/low-noise dataset",
            10, 120, 0.07, 32, 0.03, 400, 0.05, baseSeed));
        experiments.add(new ExperimentConfig(
            "BATCH_WIDE", "Concept: Batch Size", "Large batch stabilizes updates",
            12, 120, 0.08, 128, 0.02, 400, 0.20, baseSeed));
        
        // Highlighted best configuration
        experiments.add(new ExperimentConfig(
            "BEST", "Concept: Best Config", "Balanced capacity + regularization",
            20, 150, 0.04, 64, 0.08, 800, 0.15, baseSeed));
        
        return experiments;
    }
    
    /**
     * Run a single experiment and return results
     */
    private ExperimentResult runExperiment(ExperimentConfig config) {
        long startTime = System.currentTimeMillis();
        
        // Initialize RNGs
        var initRng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(config.seed);
        var dataRng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(config.seed ^ 0x9E3779B97F4A7C15L);
        
        // Generate dataset
        Datasets.MoonsData data = Datasets.makeMoons(config.samples, config.noise, config.seed);
        
        // Train/test split (80/20)
        int trainSize = (int) (config.samples * 0.8);
        
        Tensor X_train = extractRows(data.X(), 0, trainSize);
        Tensor y_train = extractRows(data.y(), 0, trainSize);
        Tensor X_test = extractRows(data.X(), trainSize, config.samples);
        Tensor y_test = extractRows(data.y(), trainSize, config.samples);
        
        // Create model
        VectorMLP model = new VectorMLP(2, new int[]{config.hidden}, 1, true, initRng);
        
        // Create optimizer
        SGD optimizer = new SGD(config.lr);
        
        // Track history
        List<Double> trainAccHistory = new ArrayList<>();
        List<Double> testAccHistory = new ArrayList<>();
        List<Double> lossHistory = new ArrayList<>();
        
        // Training loop
        for (int epoch = 0; epoch < config.epochs; epoch++) {
            // Shuffle training data
            int[] indices = shuffle(trainSize, dataRng);
            
            double epochLoss = 0.0;
            int numBatches = 0;
            
            // Mini-batch training
            for (int start = 0; start < trainSize; start += config.batchSize) {
                int end = Math.min(start + config.batchSize, trainSize);
                int[] batchIndices = Arrays.copyOfRange(indices, start, end);
                
                Tensor X_batch = extractRowsByIndices(X_train, batchIndices);
                Tensor y_batch = extractRowsByIndices(y_train, batchIndices);
                
                // Forward
                Tensor score = model.forward(X_batch);
                Tensor loss = Losses.hingeLossWithL2(score, y_batch, model.weights(), config.l2);
                
                // Backward
                Tensor.backward(loss);
                
                // Update
                optimizer.step(model.parameters());
                model.zeroGrad();
                
                epochLoss += loss.item();
                numBatches++;
            }
            
            epochLoss /= numBatches;
            lossHistory.add(epochLoss);
            
            // Evaluate every 10 epochs or at the end
            if ((epoch + 1) % 10 == 0 || epoch == config.epochs - 1) {
                double trainAcc = evaluate(model, X_train, y_train);
                double testAcc = evaluate(model, X_test, y_test);
                trainAccHistory.add(trainAcc);
                testAccHistory.add(testAcc);
            }
        }
        
        // Final evaluation
        double finalTrainAcc = evaluate(model, X_train, y_train);
        double finalTestAcc = evaluate(model, X_test, y_test);
        double finalLoss = lossHistory.get(lossHistory.size() - 1);
        
        long durationMs = System.currentTimeMillis() - startTime;
        
        return new ExperimentResult(config, finalTrainAcc, finalTestAcc, finalLoss,
                                   trainAccHistory, testAccHistory, lossHistory, durationMs);
    }
    
    /**
     * Main test that runs all experiments and generates report
     */
    @Test
    void runAllExperiments() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("EXPERIMENTAL LEARNING TEST SUITE");
        System.out.println("=".repeat(80) + "\n");
        
        List<ExperimentConfig> experiments = defineExperiments();
        List<ExperimentResult> results = new ArrayList<>();
        
        System.out.printf("Running %d experiments...%n%n", experiments.size());
        
        // Run all experiments
        for (int i = 0; i < experiments.size(); i++) {
            ExperimentConfig config = experiments.get(i);
            System.out.printf("[%d/%d] Running: %s - %s%n", 
                            i + 1, experiments.size(), config.name, config.concept);
            
            try {
                ExperimentResult result = runExperiment(config);
                results.add(result);
                
                System.out.printf("  ✓ Train Acc: %.4f | Test Acc: %.4f | Gap: %.4f | Loss: %.4f | Time: %dms%n",
                                result.trainAccuracy, result.testAccuracy, result.accuracyGap,
                                result.finalLoss, result.durationMs);
            } catch (Exception e) {
                System.err.printf("  ✗ Failed: %s%n", e.getMessage());
                e.printStackTrace();
            }
        }
        
        // Generate report
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        Path reportDir = Path.of("runs", "experimental-learning-" + timestamp);
        try {
            Files.createDirectories(reportDir);
            generateReport(results, reportDir);
        } catch (IOException e) {
            System.err.println("Failed to generate report: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("All experiments completed!");
        System.out.printf("Report saved to: %s%n", reportDir.toAbsolutePath());
        System.out.println("=".repeat(80) + "\n");
    }
    
    /**
     * Generate comprehensive report
     */
    private void generateReport(List<ExperimentResult> results, Path reportDir) throws IOException {
        ExperimentResult best = results.stream()
            .max(Comparator.comparing(r -> r.testAccuracy))
            .orElse(null);
        
        // Generate summary report
        Path summaryPath = reportDir.resolve("EXPERIMENT_SUMMARY.md");
        try (FileWriter writer = new FileWriter(summaryPath.toFile())) {
            writer.write("# Neural Concept Experiments\n\n");
            writer.write("Generated: " + LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + "\n\n");
            writer.write(String.format("- Total Experiments: %d%n", results.size()));
            long underfitCount = results.stream().filter(r -> "Underfitting".equals(classifyBehavior(r))).count();
            long overfitCount = results.stream().filter(r -> "Overfitting".equals(classifyBehavior(r))).count();
            writer.write(String.format("- Detected Underfitting cases: %d%n", underfitCount));
            writer.write(String.format("- Detected Overfitting cases: %d%n%n", overfitCount));
            
            if (best != null) {
                writer.write("## Highlight: Best Configuration\n");
                writer.write(String.format("**%s (%s)**%n%n", best.config.name, best.config.concept));
                writer.write(String.format("- Train/Test Accuracy: %.3f / %.3f%n", best.trainAccuracy, best.testAccuracy));
                writer.write(String.format("- Gap: %.3f | Behavior: %s%n", best.accuracyGap, classifyBehavior(best)));
                writer.write(String.format("- Hyperparameters: hidden=%d, epochs=%d, lr=%.3f, l2=%.2f, batch=%d, samples=%d, noise=%.2f%n%n",
                    best.config.hidden, best.config.epochs, best.config.lr, best.config.l2,
                    best.config.batchSize, best.config.samples, best.config.noise));
            }
            
            // Summary table with behavior labels
            writer.write("## Results Summary\n\n");
            writer.write("| ID | Concept | Hidden | L2 | Epochs | Train Acc | Test Acc | Gap | Behavior | Time (ms) |\n");
            writer.write("|----|---------|--------|----|--------|-----------|----------|-----|----------|-----------|\n");
            for (ExperimentResult result : results) {
                ExperimentConfig config = result.config;
                writer.write(String.format("| %s | %s | %d | %.2f | %d | %.4f | %.4f | %.4f | %s | %d |%n",
                    config.name, config.concept, config.hidden, config.l2, config.epochs,
                    result.trainAccuracy, result.testAccuracy, result.accuracyGap,
                    classifyBehavior(result), result.durationMs));
            }
            
            // Concept sections
            writer.write("\n## Concept Walkthrough\n\n");
            Map<String, List<ExperimentResult>> byPhase = new LinkedHashMap<>();
            for (ExperimentResult result : results) {
                byPhase.computeIfAbsent(result.config.phase, k -> new ArrayList<>()).add(result);
            }
            
            for (Map.Entry<String, List<ExperimentResult>> entry : byPhase.entrySet()) {
                writer.write("### " + entry.getKey() + "\n\n");
                for (ExperimentResult result : entry.getValue()) {
                    writer.write(String.format("**%s** — %s%n", result.config.name, result.config.concept));
                    writer.write(String.format("- Train/Test: %.3f / %.3f (gap %.3f)%n", result.trainAccuracy, result.testAccuracy, result.accuracyGap));
                    writer.write(String.format("- Behavior: %s%n", classifyBehavior(result)));
                    writer.write(String.format("- Loss: %.4f | Duration: %d ms%n", result.finalLoss, result.durationMs));
                    writer.write(String.format("- Hyperparameters: hidden=%d, epochs=%d, lr=%.3f, l2=%.2f, batch=%d, samples=%d, noise=%.2f%n%n",
                        result.config.hidden, result.config.epochs, result.config.lr, result.config.l2,
                        result.config.batchSize, result.config.samples, result.config.noise));
                }
            }
        }
        
        // Generate CSV for easy analysis
        Path csvPath = reportDir.resolve("experiment_results.csv");
        try (FileWriter writer = new FileWriter(csvPath.toFile())) {
            writer.write("Experiment,Phase,Concept,Hidden,Epochs,LR,Batch,L2,Samples,Noise,");
            writer.write("TrainAcc,TestAcc,Gap,Behavior,FinalLoss,DurationMs\n");
            
            for (ExperimentResult result : results) {
                ExperimentConfig config = result.config;
                writer.write(String.format("%s,\"%s\",\"%s\",%d,%d,%.2f,%d,%.2f,%d,%.2f,",
                    config.name, config.phase, config.concept, config.hidden, config.epochs,
                    config.lr, config.batchSize, config.l2, config.samples, config.noise));
                writer.write(String.format("%.4f,%.4f,%.4f,%s,%.4f,%d%n",
                    result.trainAccuracy, result.testAccuracy, result.accuracyGap,
                    classifyBehavior(result), result.finalLoss, result.durationMs));
            }
        } catch (IOException e) {
            System.err.println("Failed to write CSV report: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("Generated reports:");
        System.out.println("  - " + summaryPath.toAbsolutePath());
        System.out.println("  - " + csvPath.toAbsolutePath());
        
        generateVisualizations(results, reportDir, best);
    }
    
    private String classifyBehavior(ExperimentResult result) {
        double train = result.trainAccuracy;
        double test = result.testAccuracy;
        double gap = result.accuracyGap;
        
        if (train < 0.75 && test < 0.65) {
            return "Underfitting";
        }
        if (gap > 0.20 && test < 0.8) {
            return "Overfitting";
        }
        if (test >= 0.85 && Math.abs(gap) <= 0.10) {
            return "Well-balanced";
        }
        return "Mixed";
    }
    
    private void generateVisualizations(List<ExperimentResult> results, Path reportDir, ExperimentResult best) {
        if (results.isEmpty()) {
            return;
        }
        try {
            List<String> labels = new ArrayList<>();
            List<Double> trainData = new ArrayList<>();
            List<Double> testData = new ArrayList<>();
            List<Double> gapData = new ArrayList<>();
            
            for (ExperimentResult r : results) {
                labels.add(r.config.name);
                trainData.add(r.trainAccuracy);
                testData.add(r.testAccuracy);
                gapData.add(r.accuracyGap);
            }
            
            CategoryChart accuracyChart = new CategoryChartBuilder()
                .width(1100)
                .height(600)
                .title("Train vs Test Accuracy")
                .xAxisTitle("Experiment")
                .yAxisTitle("Accuracy")
                .build();
            accuracyChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
            accuracyChart.getStyler().setXAxisLabelRotation(45);
            accuracyChart.addSeries("Train", labels, trainData);
            accuracyChart.addSeries("Test", labels, testData);
            BitmapEncoder.saveBitmap(accuracyChart,
                reportDir.resolve("accuracy_comparison").toString(),
                BitmapFormat.PNG);
            
            CategoryChart gapChart = new CategoryChartBuilder()
                .width(1100)
                .height(500)
                .title("Accuracy Gap (Train - Test)")
                .xAxisTitle("Experiment")
                .yAxisTitle("Gap")
                .build();
            gapChart.getStyler().setXAxisLabelRotation(45);
            gapChart.addSeries("Gap", labels, gapData);
            BitmapEncoder.saveBitmap(gapChart,
                reportDir.resolve("accuracy_gap").toString(),
                BitmapFormat.PNG);
            
            if (best != null && best.lossHistory.size() > 1) {
                List<Integer> epochs = new ArrayList<>();
                for (int i = 0; i < best.lossHistory.size(); i++) {
                    epochs.add(i + 1);
                }
                XYChart lossChart = new XYChartBuilder()
                    .width(900)
                    .height(500)
                    .title("Loss Trend for " + best.config.name)
                    .xAxisTitle("Epoch")
                    .yAxisTitle("Loss")
                    .build();
                lossChart.addSeries("Loss", epochs, best.lossHistory);
                BitmapEncoder.saveBitmap(lossChart,
                    reportDir.resolve("loss_trend_" + best.config.name).toString(),
                    BitmapFormat.PNG);
            }
        } catch (IOException e) {
            System.err.println("Failed to save visualizations: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Helper methods (same as MoonsPipelineTest)
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
    
    private int[] shuffle(int n, java.util.random.RandomGenerator rng) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
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
}

