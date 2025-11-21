package ai.micrograd.demo;

/**
 * Main CLI dispatcher for all demos and training.
 * 
 * <p>Routes commands to appropriate demo classes:
 * <ul>
 *   <li>tensor-basics → TensorBasicsDemo</li>
 *   <li>tensor-ops-demo → TensorOpsDemo</li>
 *   <li>single-step-demo → SingleStepDemo</li>
 *   <li>--flags or no args → MoonsBinaryClassifier (training)</li>
 * </ul>
 * 
 * @author Vaibhav Khare
 */
public final class Main {
    
    public static void main(String[] args) throws Exception {
        // If no args or first arg starts with --, delegate to training
        if (args.length == 0 || args[0].startsWith("--")) {
            MoonsBinaryClassifier.main(args);
            return;
        }
        
        // Otherwise, dispatch to demo
        String cmd = args[0];
        String[] rest = java.util.Arrays.copyOfRange(args, 1, args.length);
        
        switch (cmd) {
            case "tensor-basics" -> TensorBasicsDemo.main(rest);
            case "tensor-ops-demo" -> TensorOpsDemo.main(rest);
            case "single-step-demo" -> SingleStepDemo.main(rest);
            default -> {
                System.err.println("Unknown command: " + cmd);
                System.err.println("\nAvailable commands:");
                System.err.println("  tensor-basics       - Basic tensor creation and operations");
                System.err.println("  tensor-ops-demo     - Tensor operations showcase");
                System.err.println("  single-step-demo    - Single forward/backward pass demo");
                System.err.println("  [training flags]    - Train binary classifier (default)");
                System.err.println("\nExamples:");
                System.err.println("  ./gradlew run --args=\"tensor-basics\"");
                System.err.println("  ./gradlew run --args=\"--hidden 4 --epochs 200 --lr 0.1\"");
                System.exit(2);
            }
        }
    }
}

