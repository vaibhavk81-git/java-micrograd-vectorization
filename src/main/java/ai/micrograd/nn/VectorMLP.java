package ai.micrograd.nn;

import ai.micrograd.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * VectorMLP is a multi-layer perceptron for vectorized (batched) inputs.
 * 
 * <p>Constructs a stack of Linear layers with activations between hidden layers.
 * The output layer has no activation (linear output).
 * 
 * <p><b>Architecture:</b>
 * <pre>
 * Input → Linear → Activation → Linear → Activation → ... → Linear → Output
 * </pre>
 * 
 * @author Vaibhav Khare
 */
public final class VectorMLP {
    
    private final Linear[] layers;
    private final boolean useTanh;
    
    /**
     * Creates a vectorized MLP.
     * 
     * @param inputDim input dimension
     * @param hidden array of hidden layer dimensions
     * @param outputDim output dimension
     * @param useTanh if true, use tanh activation; if false, use ReLU
     * @param initRng random generator for parameter initialization
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public VectorMLP(int inputDim, int[] hidden, int outputDim, boolean useTanh, RandomGenerator initRng) {
        if (inputDim < 1) {
            throw new IllegalArgumentException("inputDim must be at least 1, got: " + inputDim);
        }
        if (outputDim < 1) {
            throw new IllegalArgumentException("outputDim must be at least 1, got: " + outputDim);
        }
        if (hidden == null) {
            throw new IllegalArgumentException("hidden cannot be null");
        }
        if (initRng == null) {
            throw new IllegalArgumentException("initRng cannot be null");
        }
        
        this.useTanh = useTanh;
        this.layers = new Linear[hidden.length + 1];
        
        int prev = inputDim;
        for (int i = 0; i < hidden.length; i++) {
            if (hidden[i] < 1) {
                throw new IllegalArgumentException(
                    String.format("hidden[%d] must be at least 1, got: %d", i, hidden[i]));
            }
            layers[i] = new Linear(prev, hidden[i], initRng);
            prev = hidden[i];
        }
        
        layers[layers.length - 1] = new Linear(prev, outputDim, initRng);
    }
    
    /**
     * Forward pass through the network.
     * 
     * @param X input tensor (batch_size × input_dim)
     * @return output tensor (batch_size × output_dim)
     */
    public Tensor forward(Tensor X) {
        Tensor out = X;
        
        // Hidden layers with activation
        for (int i = 0; i < layers.length - 1; i++) {
            out = layers[i].forward(out);
            out = useTanh ? Activations.tanh(out) : Activations.relu(out);
        }
        
        // Output layer (no activation)
        out = layers[layers.length - 1].forward(out);
        
        return out;
    }
    
    /**
     * Returns all learnable parameters from all layers.
     * 
     * @return list of all parameter tensors
     */
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        for (Linear layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }
    
    /**
     * Returns only weight parameters (no biases) from all layers.
     * Useful for L2 regularization.
     * 
     * @return list of weight tensors
     */
    public List<Tensor> weights() {
        List<Tensor> weights = new ArrayList<>();
        for (Linear layer : layers) {
            weights.addAll(layer.weights());
        }
        return weights;
    }
    
    /**
     * Zeros gradients for all parameters.
     */
    public void zeroGrad() {
        for (Tensor param : parameters()) {
            param.zeroGrad();
        }
    }
    
    public int numLayers() {
        return layers.length;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("VectorMLP(");
        for (int i = 0; i < layers.length; i++) {
            if (i > 0) sb.append(" → ");
            sb.append(layers[i].inFeatures());
        }
        sb.append(" → ").append(layers[layers.length - 1].outFeatures());
        sb.append(", activation=").append(useTanh ? "tanh" : "relu");
        sb.append(")");
        return sb.toString();
    }
}

