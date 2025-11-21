package ai.micrograd.optim;

import ai.micrograd.tensor.Tensor;
import java.util.List;

/**
 * SGD (Stochastic Gradient Descent) optimizer.
 * 
 * <p>Performs vanilla gradient descent updates:
 * <pre>
 * param.data -= learning_rate * param.grad
 * </pre>
 * 
 * <p>This is the simplest optimizer and serves as a baseline.
 * Future enhancements could include momentum, Adam, etc.
 * 
 * @author Vaibhav Khare
 */
public final class SGD {
    
    private final double lr;
    
    /**
     * Creates an SGD optimizer with the specified learning rate.
     * 
     * @param lr learning rate (step size)
     * @throws IllegalArgumentException if lr is not positive
     */
    public SGD(double lr) {
        if (lr <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive, got: " + lr);
        }
        this.lr = lr;
    }
    
    /**
     * Performs a single optimization step on one parameter.
     * 
     * <p>Updates in-place: param.data[i] -= lr * param.grad[i]
     * 
     * @param param parameter tensor to update
     * @throws IllegalStateException if param does not require gradients
     */
    public void step(Tensor param) {
        if (!param.requiresGrad()) {
            throw new IllegalStateException("Cannot step on tensor without gradients");
        }
        
        double[] data = param.data();
        double[] grad = param.grad();
        
        for (int i = 0; i < data.length; i++) {
            data[i] -= lr * grad[i];
        }
    }
    
    /**
     * Performs a single optimization step on all parameters.
     * 
     * @param parameters list of parameter tensors to update
     */
    public void step(List<Tensor> parameters) {
        for (Tensor param : parameters) {
            if (param.requiresGrad()) {
                step(param);
            }
        }
    }
    
    /**
     * Zeros gradients for all parameters.
     * 
     * <p>Convenience method to avoid calling zeroGrad() on each parameter individually.
     * 
     * @param parameters list of parameter tensors
     */
    public static void zeroGrad(List<Tensor> parameters) {
        for (Tensor param : parameters) {
            if (param.requiresGrad()) {
                param.zeroGrad();
            }
        }
    }
    
    public double getLearningRate() {
        return lr;
    }
    
    @Override
    public String toString() {
        return String.format("SGD(lr=%.4f)", lr);
    }
}

