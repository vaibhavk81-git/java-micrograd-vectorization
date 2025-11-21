package ai.micrograd.demo;

import ai.micrograd.data.Datasets;
import ai.micrograd.nn.Losses;
import ai.micrograd.nn.VectorMLP;
import ai.micrograd.tensor.Tensor;
import ai.micrograd.visualization.TensorGraphExporter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.random.RandomGenerator;

/**
 * SingleStepDemo demonstrates a single forward-backward pass.
 * 
 * <p>This demo shows:
 * <ul>
 *   <li>Forward pass through a small network</li>
 *   <li>Loss computation</li>
 *   <li>Backward pass (gradient computation)</li>
 *   <li>Gradient inspection</li>
 *   <li>Optional computation graph export</li>
 * </ul>
 * 
 * @author Vaibhav Khare
 */
public final class SingleStepDemo {
    
    public static void main(String[] args) throws Exception {
        System.out.println("=== Single Step Demo ===\n");
        
        // Parse output directory
        String outDirStr = "runs";
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--outDir")) {
                outDirStr = args[i + 1];
                break;
            }
        }
        
        // Create output directory with timestamp
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        Path outDir = Path.of(outDirStr, timestamp);
        Files.createDirectories(outDir);
        System.out.println("Output directory: " + outDir.toAbsolutePath() + "\n");
        
        // Generate a tiny batch from moons dataset
        System.out.println("Generating 4 samples from moons dataset...");
        Datasets.MoonsData data = Datasets.makeMoons(4, 0.1, 42);
        Tensor X = data.X();
        Tensor y = data.y();
        
        System.out.println("X shape: (" + X.rows() + ", " + X.cols() + ")");
        System.out.println("y shape: (" + y.rows() + ", " + y.cols() + ")");
        System.out.println("\nX (input features):");
        printTensor(X);
        System.out.println("\ny (labels):");
        printTensor(y);
        
        // Create a small model: 2 → 3 → 1 with tanh
        System.out.println("\n--- Creating Model ---");
        RandomGenerator rng = RandomGenerator.of("L64X128MixRandom");
        VectorMLP model = new VectorMLP(2, new int[]{3}, 1, true, rng);
        System.out.println(model);
        System.out.println("Parameters: " + model.parameters().size());
        
        // Forward pass
        System.out.println("\n--- Forward Pass ---");
        Tensor score = model.forward(X);
        System.out.println("Score shape: (" + score.rows() + ", " + score.cols() + ")");
        System.out.println("Scores:");
        printTensor(score);
        
        // Compute loss
        System.out.println("\n--- Loss Computation ---");
        double l2 = 0.01;
        Tensor loss = Losses.hingeLossWithL2(score, y, model.weights(), l2);
        System.out.println("Loss shape: (" + loss.rows() + ", " + loss.cols() + ")");
        System.out.printf("Loss value: %.6f%n", loss.item());
        
        // Backward pass
        System.out.println("\n--- Backward Pass ---");
        Tensor.backward(loss);
        System.out.println("Gradients computed!");
        
        // Inspect gradients
        System.out.println("\n--- Gradient Inspection ---");
        int paramIdx = 0;
        for (Tensor param : model.parameters()) {
            String label = param.label() != null ? param.label() : "param_" + paramIdx;
            System.out.println("\n" + label + " shape: (" + param.rows() + ", " + param.cols() + ")");
            
            // Show first few gradient values
            double[] grads = param.gradToArray();
            int showCount = Math.min(6, grads.length);
            System.out.print("First " + showCount + " gradient values: [");
            for (int i = 0; i < showCount; i++) {
                System.out.printf("%.4f", grads[i]);
                if (i < showCount - 1) System.out.print(", ");
            }
            System.out.println("]");
            
            paramIdx++;
        }
        
        // Export computation graph
        System.out.println("\n--- Exporting Computation Graph ---");
        Path dotFile = outDir.resolve("graph.dot");
        Path pngFile = outDir.resolve("graph.png");
        TensorGraphExporter.exportDotAndMaybePng(loss, dotFile, pngFile);
        
        System.out.println("\n✓ Single step demo complete!");
        System.out.println("Output saved to: " + outDir.toAbsolutePath());
    }
    
    private static void printTensor(Tensor t) {
        System.out.println("[");
        for (int r = 0; r < t.rows(); r++) {
            System.out.print("  [");
            for (int c = 0; c < t.cols(); c++) {
                System.out.printf("%7.3f", t.get(r, c));
                if (c < t.cols() - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("]");
        }
        System.out.println("]");
    }
}

