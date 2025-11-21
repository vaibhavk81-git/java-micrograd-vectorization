package ai.micrograd.nn;

/**
 * Activation functions supported by neurons in the neural network.
 * 
 * @author Vaibhav Khare
 */
public enum ActivationType {
    
    /**
     * Hyperbolic Tangent activation function.
     * Range: (-1, 1)
     */
    TANH("tanh"),
    
    /**
     * Rectified Linear Unit activation function.
     * Range: [0, ∞)
     */
    RELU("relu"),
    
    /**
     * Linear (identity) activation function.
     * Range: (-∞, ∞)
     */
    LINEAR("linear");
    
    private final String displayName;
    
    ActivationType(String displayName) {
        this.displayName = displayName;
    }
    
    public String getDisplayName() {
        return displayName;
    }
    
    public static ActivationType fromString(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Activation type name cannot be null or empty");
        }
        
        return switch (name.toLowerCase().trim()) {
            case "tanh" -> TANH;
            case "relu" -> RELU;
            case "linear", "none" -> LINEAR;
            default -> throw new IllegalArgumentException(
                String.format("Unknown activation type: '%s'. Supported values: tanh, relu, linear, none", name));
        };
    }
    
    @Override
    public String toString() {
        return displayName;
    }
}

