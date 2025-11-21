package ai.micrograd.nn;

import ai.micrograd.autograd.Value;
import java.util.Collections;
import java.util.List;

/**
 * Module is the base interface for all neural network components.
 * 
 * <p>This interface is inspired by PyTorch's nn.Module and provides a common
 * contract for all components that have learnable parameters.
 * 
 * @author Vaibhav Khare
 */
public interface Module {
    
    /**
     * Returns all learnable parameters in this module and its sub-modules.
     *
     * @return an immutable list of all Value parameters in this module
     */
    List<Value> parameters();
    
    /**
     * Zeros out the gradients of all parameters in this module.
     */
    default void zeroGrad() {
        for (Value param : parameters()) {
            param.setGrad(0.0);
        }
    }
    
    /**
     * Returns the number of learnable parameters in this module.
     *
     * @return the number of parameters in this module
     */
    default int numParameters() {
        return parameters().size();
    }
    
    /**
     * Returns an empty list of parameters.
     *
     * @return an empty immutable list
     */
    static List<Value> emptyParameters() {
        return Collections.emptyList();
    }
}

