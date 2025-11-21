package ai.micrograd.nn;

import ai.micrograd.autograd.Value;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Neuron represents a single artificial neuron in a neural network.
 * 
 * @author Vaibhav Khare
 */
public class Neuron implements Module {

    private final List<Value> weights;
    private final Value bias;
    private final ActivationType activationType;

    public Neuron(int nin) {
        this(nin, ActivationType.TANH);
    }

    public Neuron(int nin, ActivationType activationType) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }

        this.activationType = activationType;
        this.weights = new ArrayList<>(nin);
        
        Random random = new Random();
        for (int i = 0; i < nin; i++) {
            double weight = (random.nextDouble() * 2.0) - 1.0;
            this.weights.add(new Value(weight, "w" + i));
        }
        
        double biasValue = (random.nextDouble() * 2.0) - 1.0;
        this.bias = new Value(biasValue, "b");
    }

    public Neuron(int nin, ActivationType activationType, Random rng) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator cannot be null");
        }

        this.activationType = activationType;
        this.weights = new ArrayList<>(nin);

        for (int i = 0; i < nin; i++) {
            double weight = (rng.nextDouble() * 2.0) - 1.0;
            this.weights.add(new Value(weight, "w" + i));
        }

        double biasValue = (rng.nextDouble() * 2.0) - 1.0;
        this.bias = new Value(biasValue, "b");
    }

    public Neuron(int nin, Random rng) {
        this(nin, ActivationType.TANH, rng);
    }

    public Value forward(List<Value> inputs) {
        if (inputs.size() != weights.size()) {
            throw new IllegalArgumentException(
                String.format("Expected %d inputs, got %d", weights.size(), inputs.size()));
        }

        Value activation = bias;
        
        for (int i = 0; i < weights.size(); i++) {
            Value weightedInput = weights.get(i).mul(inputs.get(i));
            activation = activation.add(weightedInput);
        }
        
        if (activationType != ActivationType.LINEAR) {
            activation = applyActivation(activation);
        }
        
        return activation;
    }

    private Value applyActivation(Value value) {
        return switch (activationType) {
            case TANH -> value.tanh();
            case RELU -> value.relu();
            case LINEAR -> value;
        };
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(weights.size() + 1);
        params.addAll(weights);
        params.add(bias);
        return Collections.unmodifiableList(params);
    }

    public int getInputSize() {
        return weights.size();
    }

    public List<Value> getWeights() {
        return Collections.unmodifiableList(weights);
    }

    public Value getBias() {
        return bias;
    }

    public ActivationType getActivationType() {
        return activationType;
    }

    @Override
    public String toString() {
        return String.format("Neuron(nin=%d, activation=%s)", 
            weights.size(), 
            activationType.getDisplayName());
    }
}

