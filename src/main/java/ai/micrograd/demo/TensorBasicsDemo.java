package ai.micrograd.demo;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;

import java.util.random.RandomGenerator;

/**
 * TensorBasicsDemo demonstrates basic tensor creation and operations.
 * 
 * <p>This demo shows:
 * <ul>
 *   <li>Creating tensors from arrays</li>
 *   <li>Tensor shapes</li>
 *   <li>Pretty-printed tensor grids</li>
 *   <li>Simple matrix multiplication</li>
 * </ul>
 * 
 * @author Vaibhav Khare
 */
public final class TensorBasicsDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Tensor Basics Demo ===\n");
        
        // Create a 2×3 tensor
        System.out.println("Creating a 2×3 tensor:");
        double[][] data1 = {
            {1.5, 2.3, 3.7},
            {4.2, 5.8, 6.1}
        };
        Tensor t1 = Tensor.fromArray(data1, false);
        System.out.println("Shape: (" + t1.rows() + ", " + t1.cols() + ")");
        printTensor(t1);
        
        // Create a 3×1 tensor
        System.out.println("\nCreating a 3×1 tensor:");
        double[][] data2 = {
            {0.5},
            {1.2},
            {0.8}
        };
        Tensor t2 = Tensor.fromArray(data2, false);
        System.out.println("Shape: (" + t2.rows() + ", " + t2.cols() + ")");
        printTensor(t2);
        
        // Matrix multiplication
        System.out.println("\nMatrix multiplication (2×3) @ (3×1):");
        Tensor result = TensorOps.matmul(t1, t2);
        System.out.println("Result shape: (" + result.rows() + ", " + result.cols() + ")");
        printTensor(result);
        
        // Random tensor
        System.out.println("\nRandom 3×2 tensor:");
        RandomGenerator rng = RandomGenerator.of("L64X128MixRandom");
        Tensor t3 = Tensor.rand(3, 2, rng, -1.0, 1.0, false);
        System.out.println("Shape: (" + t3.rows() + ", " + t3.cols() + ")");
        printTensor(t3);
        
        System.out.println("\n✓ Tensor basics demo complete!");
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

