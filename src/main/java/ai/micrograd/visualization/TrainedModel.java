package ai.micrograd.visualization;

/**
 * TrainedModel provides a simple interface for querying model predictions.
 * 
 * <p>Used by visualization tools like DecisionBoundaryPlotter to evaluate
 * the model at arbitrary points in the input space.
 * 
 * @author Vaibhav Khare
 */
public interface TrainedModel {
    
    /**
     * Computes the model's score for a 2D input point.
     * 
     * @param x1 first feature
     * @param x2 second feature
     * @return model score (typically positive for class +1, negative for class -1)
     */
    double score(double x1, double x2);
}


