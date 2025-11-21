package ai.micrograd.nn;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;
import java.util.List;

/**
 * Losses provides loss functions for training neural networks.
 * 
 * @author Vaibhav Khare
 */
public final class Losses {
    
    private Losses() {} // Prevent instantiation
    
    /**
     * Computes hinge loss with L2 regularization.
     * 
     * <p><b>Hinge Loss:</b> mean(max(0, 1 - y * score))
     * where y ∈ {-1, +1} and score is the model output.
     * 
     * <p><b>L2 Regularization:</b> lambda * sum(W²) over weight matrices only (no bias).
     * 
     * <p><b>Total Loss:</b> hinge_loss + l2_penalty
     * 
     * @param score model predictions (batch_size × 1)
     * @param y true labels (batch_size × 1) with values in {-1, +1}
     * @param weights list of weight tensors to regularize (typically from Linear.weights())
     * @param lambda L2 regularization strength
     * @return scalar loss tensor
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public static Tensor hingeLossWithL2(Tensor score, Tensor y, List<Tensor> weights, double lambda) {
        if (score.rows() != y.rows() || score.cols() != 1 || y.cols() != 1) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch: score (%d, %d), y (%d, %d). Both must be (n, 1)",
                    score.rows(), score.cols(), y.rows(), y.cols()));
        }
        
        // Compute margin: 1 - y * score
        Tensor margin = TensorOps.mul(y, score);  // y * score
        
        // Create tensor of ones for subtraction
        Tensor ones = Tensor.ones(margin.rows(), margin.cols(), margin.requiresGrad());
        
        // margin = 1 - (y * score)
        margin = TensorOps.add(ones, negate(margin));
        
        // Hinge: max(0, margin)
        Tensor hinge = TensorOps.relu(margin);
        
        // Mean hinge loss
        Tensor hingeLoss = TensorOps.mean(hinge, 0);  // (1×1)
        if (hingeLoss.cols() > 1) {
            // Further reduce to scalar
            hingeLoss = TensorOps.mean(hingeLoss, 1);
        }
        
        // L2 regularization: lambda * sum(W²)
        Tensor l2Penalty = null;
        if (lambda > 0 && weights != null && !weights.isEmpty()) {
            for (Tensor W : weights) {
                Tensor W2 = TensorOps.mul(W, W);  // W²
                Tensor sumW2 = TensorOps.sum(W2, 0);  // Sum over rows
                sumW2 = TensorOps.sum(sumW2, 1);  // Sum over cols -> scalar
                
                if (l2Penalty == null) {
                    l2Penalty = sumW2;
                } else {
                    l2Penalty = TensorOps.add(l2Penalty, sumW2);
                }
            }
            
            // Scale by lambda
            Tensor lambdaTensor = Tensor.ones(1, 1, l2Penalty.requiresGrad());
            lambdaTensor.data()[0] = lambda;
            l2Penalty = TensorOps.mul(lambdaTensor, l2Penalty);
        }
        
        // Total loss
        if (l2Penalty != null) {
            return TensorOps.add(hingeLoss, l2Penalty);
        } else {
            return hingeLoss;
        }
    }
    
    /**
     * Helper to negate a tensor (multiply by -1).
     */
    private static Tensor negate(Tensor x) {
        Tensor negOne = Tensor.ones(x.rows(), x.cols(), x.requiresGrad());
        negOne.data()[0] = -1.0;
        for (int i = 1; i < negOne.data().length; i++) {
            negOne.data()[i] = -1.0;
        }
        return TensorOps.mul(negOne, x);
    }
}

