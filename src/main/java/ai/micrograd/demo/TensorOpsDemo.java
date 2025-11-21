package ai.micrograd.demo;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;

/**
 * TensorOpsDemo demonstrates tensor operations.
 * 
 * <p>This demo shows:
 * <ul>
 *   <li>Element-wise addition</li>
 *   <li>Element-wise multiplication</li>
 *   <li>Activation functions (tanh, relu)</li>
 *   <li>Matrix multiplication</li>
 *   <li>Shape transformations</li>
 * </ul>
 * 
 * @author Vaibhav Khare
 */
public final class TensorOpsDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Tensor Operations Demo ===\n");
        
        // Create test tensors
        System.out.println("Creating test tensors A and B (2×2):");
        double[][] dataA = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        double[][] dataB = {
            {0.5, 1.5},
            {2.5, 3.5}
        };
        Tensor A = Tensor.fromArray(dataA, false);
        Tensor B = Tensor.fromArray(dataB, false);
        
        System.out.println("A:");
        printTensor(A);
        System.out.println("\nB:");
        printTensor(B);
        
        // Element-wise addition
        System.out.println("\n--- Element-wise Addition: A + B ---");
        Tensor sum = TensorOps.add(A, B);
        System.out.println("Shape: (" + sum.rows() + ", " + sum.cols() + ")");
        printTensor(sum);
        
        // Element-wise multiplication
        System.out.println("\n--- Element-wise Multiplication: A * B ---");
        Tensor prod = TensorOps.mul(A, B);
        System.out.println("Shape: (" + prod.rows() + ", " + prod.cols() + ")");
        printTensor(prod);
        
        // Tanh activation
        System.out.println("\n--- Tanh Activation: tanh(A) ---");
        Tensor tanhA = TensorOps.tanh(A);
        System.out.println("Shape: (" + tanhA.rows() + ", " + tanhA.cols() + ")");
        printTensor(tanhA);
        
        // ReLU activation
        System.out.println("\n--- ReLU Activation: relu(A - 2.5) ---");
        double[][] dataC = {
            {-1.0, 2.0},
            {3.0, -0.5}
        };
        Tensor C = Tensor.fromArray(dataC, false);
        System.out.println("Input:");
        printTensor(C);
        Tensor reluC = TensorOps.relu(C);
        System.out.println("Output:");
        printTensor(reluC);
        
        // Matrix multiplication
        System.out.println("\n--- Matrix Multiplication: A @ B^T ---");
        double[][] dataD = {
            {1.0, 0.0, -1.0},
            {0.5, 1.5, 2.0}
        };
        Tensor D = Tensor.fromArray(dataD, false);
        System.out.println("A (2×2):");
        printTensor(A);
        System.out.println("\nD (2×3):");
        printTensor(D);
        
        Tensor matmulResult = TensorOps.matmul(A, D);
        System.out.println("\nA @ D^T would fail, so A @ D (2×2 @ 2×3 = 2×3):");
        System.out.println("Shape: (" + matmulResult.rows() + ", " + matmulResult.cols() + ")");
        printTensor(matmulResult);
        
        // Broadcast addition
        System.out.println("\n--- Broadcast Addition: A + rowVec ---");
        double[][] rowData = {{10.0, 20.0}};
        Tensor rowVec = Tensor.fromArray(rowData, false);
        System.out.println("Row vector (1×2):");
        printTensor(rowVec);
        Tensor broadcast = TensorOps.addRowVector(A, rowVec);
        System.out.println("\nResult (each row of A + rowVec):");
        printTensor(broadcast);
        
        System.out.println("\n✓ Tensor operations demo complete!");
    }
    
    private static void printTensor(Tensor t) {
        System.out.println("[");
        for (int r = 0; r < t.rows(); r++) {
            System.out.print("  [");
            for (int c = 0; c < t.cols(); c++) {
                System.out.printf("%6.2f", t.get(r, c));
                if (c < t.cols() - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("]");
        }
        System.out.println("]");
    }
}

