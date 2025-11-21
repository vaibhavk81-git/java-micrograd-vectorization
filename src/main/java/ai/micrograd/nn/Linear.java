package ai.micrograd.nn;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;
import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Linear (fully-connected) layer for vectorized neural networks.
 * 
 * <p>Performs the transformation: Y = X @ W + b
 * where:
 * <ul>
 *   <li>X: input (batch_size × in_features)</li>
 *   <li>W: weights (in_features × out_features)</li>
 *   <li>b: bias (1 × out_features)</li>
 *   <li>Y: output (batch_size × out_features)</li>
 * </ul>
 * 
 * <p><b>Initialization:</b> Uses Xavier uniform initialization:
 * <pre>
 * limit = sqrt(6 / (in_features + out_features))
 * W ~ Uniform(-limit, limit)
 * b ~ Uniform(-limit, limit)
 * </pre>
 * 
 * @author Vaibhav Khare
 */
public final class Linear {
    
    private final Tensor W;
    private final Tensor b;
    private final int inFeatures;
    private final int outFeatures;
    
    /**
     * Creates a linear layer with Xavier uniform initialization.
     * 
     * @param inFeatures number of input features
     * @param outFeatures number of output features
     * @param initRng random generator for parameter initialization
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public Linear(int inFeatures, int outFeatures, RandomGenerator initRng) {
        if (inFeatures < 1) {
            throw new IllegalArgumentException("inFeatures must be at least 1, got: " + inFeatures);
        }
        if (outFeatures < 1) {
            throw new IllegalArgumentException("outFeatures must be at least 1, got: " + outFeatures);
        }
        if (initRng == null) {
            throw new IllegalArgumentException("initRng cannot be null");
        }
        
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        
        // Xavier uniform initialization
        double limit = Math.sqrt(6.0 / (inFeatures + outFeatures));
        
        this.W = Tensor.rand(inFeatures, outFeatures, initRng, -limit, limit, true);
        this.W.setLabel("W");
        
        this.b = Tensor.rand(1, outFeatures, initRng, -limit, limit, true);
        this.b.setLabel("b");
    }
    
    /**
     * Forward pass: Y = X @ W + b
     * 
     * @param X input tensor (batch_size × in_features)
     * @return output tensor (batch_size × out_features)
     * @throws IllegalArgumentException if input shape is incompatible
     */
    public Tensor forward(Tensor X) {
        if (X.cols() != inFeatures) {
            throw new IllegalArgumentException(
                String.format("Input has %d features, expected %d", X.cols(), inFeatures));
        }
        
        // Y = X @ W
        Tensor Y = TensorOps.matmul(X, W);
        
        // Y = Y + b (broadcast)
        Y = TensorOps.addRowVector(Y, b);
        
        return Y;
    }
    
    /**
     * Returns all learnable parameters [W, b].
     * 
     * @return list of parameter tensors
     */
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>(2);
        params.add(W);
        params.add(b);
        return params;
    }
    
    /**
     * Returns only the weight parameters (excluding bias).
     * Useful for L2 regularization.
     * 
     * @return list containing only W
     */
    public List<Tensor> weights() {
        List<Tensor> weights = new ArrayList<>(1);
        weights.add(W);
        return weights;
    }
    
    public int inFeatures() {
        return inFeatures;
    }
    
    public int outFeatures() {
        return outFeatures;
    }
    
    public Tensor getWeight() {
        return W;
    }
    
    public Tensor getBias() {
        return b;
    }
    
    @Override
    public String toString() {
        return String.format("Linear(in_features=%d, out_features=%d)", inFeatures, outFeatures);
    }
}

