package ai.micrograd.visualization;

import ai.micrograd.nn.VectorMLP;
import ai.micrograd.tensor.Tensor;

/**
 * VectorMLPAdapter wraps a VectorMLP to implement the TrainedModel interface.
 * 
 * <p>This adapter allows visualization tools to query the model without
 * needing to know the internal tensor representation.
 * 
 * @author Vaibhav Khare
 */
public final class VectorMLPAdapter implements TrainedModel {
    
    private final VectorMLP model;
    
    /**
     * Creates an adapter for the given model.
     * 
     * @param model the vectorized MLP to wrap
     */
    public VectorMLPAdapter(VectorMLP model) {
        if (model == null) {
            throw new IllegalArgumentException("model cannot be null");
        }
        this.model = model;
    }
    
    @Override
    public double score(double x1, double x2) {
        Tensor input = new Tensor(1, 2, false);
        input.set(0, 0, x1);
        input.set(0, 1, x2);
        
        Tensor output = model.forward(input);
        return output.get(0, 0);
    }
}


