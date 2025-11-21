package ai.micrograd.nn;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;

/**
 * Activations provides activation functions for vectorized neural networks.
 * 
 * <p>These are thin wrappers around TensorOps that provide a cleaner API
 * and match the naming conventions of popular frameworks.
 * 
 * @author Vaibhav Khare
 */
public final class Activations {
    
    private Activations() {} // Prevent instantiation
    
    /**
     * Applies hyperbolic tangent activation element-wise.
     * 
     * <p>tanh(x) squashes values to range (-1, 1).
     * 
     * @param x input tensor
     * @return output tensor with same shape
     */
    public static Tensor tanh(Tensor x) {
        return TensorOps.tanh(x);
    }
    
    /**
     * Applies ReLU (Rectified Linear Unit) activation element-wise.
     * 
     * <p>ReLU(x) = max(0, x)
     * 
     * @param x input tensor
     * @return output tensor with same shape
     */
    public static Tensor relu(Tensor x) {
        return TensorOps.relu(x);
    }
}

