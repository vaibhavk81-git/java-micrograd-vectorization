package ai.micrograd.nn;

import ai.micrograd.autograd.Value;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Layer represents a collection of neurons that process the same inputs in parallel.
 * 
 * @author Vaibhav Khare
 */
public class Layer implements Module {

    private final List<Neuron> neurons;
    private final int inputSize;
    private final int outputSize;

    public Layer(int nin, int nout) {
        this(nin, nout, ActivationType.TANH);
    }

    public Layer(int nin, int nout, ActivationType activationType) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        if (nout < 1) {
            throw new IllegalArgumentException("Number of outputs must be at least 1, got: " + nout);
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }

        this.inputSize = nin;
        this.outputSize = nout;
        this.neurons = new ArrayList<>(nout);

        for (int i = 0; i < nout; i++) {
            this.neurons.add(new Neuron(nin, activationType));
        }
    }

    public Layer(int nin, int nout, ActivationType activationType, Random rng) {
        if (nin < 1) {
            throw new IllegalArgumentException("Number of inputs must be at least 1, got: " + nin);
        }
        if (nout < 1) {
            throw new IllegalArgumentException("Number of outputs must be at least 1, got: " + nout);
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator cannot be null");
        }

        this.inputSize = nin;
        this.outputSize = nout;
        this.neurons = new ArrayList<>(nout);

        for (int i = 0; i < nout; i++) {
            this.neurons.add(new Neuron(nin, activationType, rng));
        }
    }

    public Layer(int nin, int nout, Random rng) {
        this(nin, nout, ActivationType.TANH, rng);
    }

    public List<Value> forward(List<Value> inputs) {
        if (inputs.size() != inputSize) {
            throw new IllegalArgumentException(
                String.format("Expected %d inputs, got %d", inputSize, inputs.size()));
        }

        List<Value> outputs = new ArrayList<>(outputSize);
        for (Neuron neuron : neurons) {
            outputs.add(neuron.forward(inputs));
        }
        return outputs;
    }

    @Override
    public List<Value> parameters() {
        List<Value> allParams = new ArrayList<>();
        for (Neuron neuron : neurons) {
            allParams.addAll(neuron.parameters());
        }
        return Collections.unmodifiableList(allParams);
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public List<Neuron> getNeurons() {
        return Collections.unmodifiableList(neurons);
    }

    @Override
    public String toString() {
        String activation = neurons.isEmpty() ? "none" : 
            neurons.get(0).getActivationType().getDisplayName();
        return String.format("Layer(nin=%d, nout=%d, activation=%s)", 
            inputSize, outputSize, activation);
    }
}

