package ai.micrograd.nn;

import ai.micrograd.autograd.Value;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * MLP (Multi-Layer Perceptron) represents a complete feedforward neural network.
 * 
 * @author Vaibhav Khare
 */
public class MLP implements Module {

    private final List<Layer> layers;
    private final List<Integer> layerSizes;

    public MLP(List<Integer> layerSizes) {
        this(layerSizes, ActivationType.TANH);
    }

    public MLP(List<Integer> layerSizes, ActivationType activationType) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new IllegalArgumentException(
                "MLP requires at least 2 layer sizes (input and output), got: " + 
                (layerSizes == null ? "null" : layerSizes.size()));
        }

        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }

        for (int i = 0; i < layerSizes.size(); i++) {
            if (layerSizes.get(i) < 1) {
                throw new IllegalArgumentException(
                    String.format("Layer size at index %d must be positive, got: %d", 
                        i, layerSizes.get(i)));
            }
        }

        this.layerSizes = new ArrayList<>(layerSizes);
        this.layers = new ArrayList<>();

        for (int i = 0; i < layerSizes.size() - 1; i++) {
            int nin = layerSizes.get(i);
            int nout = layerSizes.get(i + 1);
            
            boolean isOutputLayer = (i == layerSizes.size() - 2);
            ActivationType layerActivationType = isOutputLayer ? ActivationType.LINEAR : activationType;
            
            layers.add(new Layer(nin, nout, layerActivationType));
        }
    }

    public MLP(List<Integer> layerSizes, ActivationType activationType, Random rng) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new IllegalArgumentException(
                "MLP requires at least 2 layer sizes (input and output), got: " + 
                (layerSizes == null ? "null" : layerSizes.size()));
        }
        if (activationType == null) {
            throw new IllegalArgumentException("Activation type cannot be null");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator cannot be null");
        }

        for (int i = 0; i < layerSizes.size(); i++) {
            if (layerSizes.get(i) < 1) {
                throw new IllegalArgumentException(
                    String.format("Layer size at index %d must be positive, got: %d", 
                        i, layerSizes.get(i)));
            }
        }

        this.layerSizes = new ArrayList<>(layerSizes);
        this.layers = new ArrayList<>();

        for (int i = 0; i < layerSizes.size() - 1; i++) {
            int nin = layerSizes.get(i);
            int nout = layerSizes.get(i + 1);

            boolean isOutputLayer = (i == layerSizes.size() - 2);
            ActivationType layerActivationType = isOutputLayer ? ActivationType.LINEAR : activationType;

            layers.add(new Layer(nin, nout, layerActivationType, rng));
        }
    }

    public MLP(List<Integer> layerSizes, Random rng) {
        this(layerSizes, ActivationType.TANH, rng);
    }

    public Value forward(List<Value> inputs) {
        List<Value> outputs = forwardAll(inputs);
        
        if (outputs.size() != 1) {
            throw new IllegalStateException(
                String.format("forward() expects single output, but network produces %d outputs. " +
                             "Use forwardAll() for multi-output networks.", outputs.size()));
        }
        
        return outputs.get(0);
    }

    public List<Value> forwardAll(List<Value> inputs) {
        if (inputs.size() != layerSizes.get(0)) {
            throw new IllegalArgumentException(
                String.format("Expected %d inputs, got %d", layerSizes.get(0), inputs.size()));
        }

        List<Value> current = inputs;
        
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        
        return current;
    }

    @Override
    public List<Value> parameters() {
        List<Value> allParams = new ArrayList<>();
        for (Layer layer : layers) {
            allParams.addAll(layer.parameters());
        }
        return Collections.unmodifiableList(allParams);
    }

    public List<Integer> getLayerSizes() {
        return Collections.unmodifiableList(layerSizes);
    }

    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    public int getNumLayers() {
        return layers.size();
    }

    public int getInputSize() {
        return layerSizes.get(0);
    }

    public int getOutputSize() {
        return layerSizes.get(layerSizes.size() - 1);
    }

    @Override
    public String toString() {
        return String.format("MLP(layers=%s, total_params=%d)", 
            layerSizes, parameters().size());
    }
}

