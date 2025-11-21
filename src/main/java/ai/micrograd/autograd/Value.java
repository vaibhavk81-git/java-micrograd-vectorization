package ai.micrograd.autograd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Value represents a node in a computational graph with automatic differentiation capabilities.
 * 
 * <p>This class implements the core autograd engine, similar to PyTorch's or TensorFlow's
 * automatic differentiation. Each Value stores:
 * <ul>
 *   <li>The actual data (forward pass result)</li>
 *   <li>The gradient (computed during backward pass)</li>
 *   <li>References to parent nodes in the computation graph</li>
 *   <li>The operation that created this Value</li>
 *   <li>A function to compute gradients during backpropagation</li>
 * </ul>
 * 
 * <p><b>⚠️ THREAD SAFETY WARNING:</b>
 * This class is <b>NOT thread-safe</b>. Each thread should use its own Value instances.
 * 
 * @author Vaibhav Khare
 */
public class Value {

    private double data;
    private double grad;
    private final List<Value> parents;
    private final String op;
    private String label;
    private Runnable backwardFn;

    /**
     * Creates a leaf Value node (typically an input or parameter).
     *
     * @param data  the numeric value
     * @param label optional label for visualization (can be empty string)
     */
    public Value(double data, String label) {
        this.data = data;
        this.grad = 0.0;
        this.parents = new ArrayList<>();
        this.op = "";
        this.label = label;
        this.backwardFn = () -> {};
    }

    /**
     * Creates a Value node resulting from an operation.
     *
     * @param data    the computed value
     * @param parents the input Values that produced this result
     * @param op      the operation name (e.g., "+", "*", "tanh")
     * @param label   optional label for visualization (can be empty string)
     */
    public Value(double data, List<Value> parents, String op, String label) {
        this.data = data;
        this.grad = 0.0;
        this.parents = new ArrayList<>(parents);
        this.op = op;
        this.label = label;
        this.backwardFn = () -> {};
    }

    public double getData() {
        return data;
    }

    public void setData(double data) {
        this.data = data;
    }
    
    public double getGrad() {
        return grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    public List<Value> getParents() {
        return Collections.unmodifiableList(parents);
    }

    public String getOp() {
        return op;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    // Arithmetic operations

    public Value add(Value other) {
        Value a = this;
        Value b = other;
        Value out = new Value(a.data + b.data, new ArrayList<>(Arrays.asList(a, b)), "+", "");
        out.backwardFn = () -> {
            a.grad += out.grad;
            b.grad += out.grad;
        };
        return out;
    }

    public Value mul(Value other) {
        Value a = this;
        Value b = other;
        Value out = new Value(a.data * b.data, new ArrayList<>(Arrays.asList(a, b)), "*", "");
        out.backwardFn = () -> {
            a.grad += b.data * out.grad;
            b.grad += a.data * out.grad;
        };
        return out;
    }

    public Value neg() {
        Value a = this;
        Value out = new Value(-a.data, new ArrayList<>(Arrays.asList(a)), "neg", "");
        out.backwardFn = () -> a.grad += -out.grad;
        return out;
    }

    public Value sub(Value other) {
        return this.add(other.neg());
    }

    public Value pow(double exponent) {
        Value a = this;
        Value out = new Value(Math.pow(a.data, exponent), new ArrayList<>(Arrays.asList(a)), "pow", "");
        out.backwardFn = () -> a.grad += exponent * Math.pow(a.data, exponent - 1) * out.grad;
        return out;
    }

    public Value div(Value other) {
        return this.mul(other.pow(-1));
    }

    // Scalar operations (for convenience)
    
    public Value add(double scalar) {
        return this.add(new Value(scalar, ""));
    }

    public Value mul(double scalar) {
        return this.mul(new Value(scalar, ""));
    }

    public Value sub(double scalar) {
        return this.sub(new Value(scalar, ""));
    }

    public Value div(double scalar) {
        return this.div(new Value(scalar, ""));
    }

    // Right-hand operations

    public Value rsub(double scalar) {
        return new Value(scalar, "").sub(this);
    }

    public Value rdiv(double scalar) {
        return new Value(scalar, "").div(this);
    }

    // Activation Functions

    public Value tanh() {
        Value a = this;
        double t = Math.tanh(a.data);
        Value out = new Value(t, new ArrayList<>(Arrays.asList(a)), "tanh", "");
        out.backwardFn = () -> a.grad += (1 - t * t) * out.grad;
        return out;
    }

    public Value exp() {
        Value a = this;
        double x = Math.exp(a.data);
        Value out = new Value(x, new ArrayList<>(Arrays.asList(a)), "exp", "");
        out.backwardFn = () -> a.grad += x * out.grad;
        return out;
    }

    public Value relu() {
        Value a = this;
        Value out = new Value(a.data < 0 ? 0 : a.data, new ArrayList<>(Arrays.asList(a)), "ReLU", "");
        out.backwardFn = () -> a.grad += (out.data > 0 ? 1 : 0) * out.grad;
        return out;
    }

    public Value sigmoid() {
        Value a = this;
        double sigmoid = 1.0 / (1.0 + Math.exp(-a.data));
        Value out = new Value(sigmoid, new ArrayList<>(Arrays.asList(a)), "sigmoid", "");
        out.backwardFn = () -> a.grad += sigmoid * (1.0 - sigmoid) * out.grad;
        return out;
    }

    public Value log() {
        Value a = this;
        if (a.data <= 0) {
            throw new ArithmeticException("Cannot compute log of non-positive number: " + a.data);
        }
        Value out = new Value(Math.log(a.data), new ArrayList<>(Arrays.asList(a)), "log", "");
        out.backwardFn = () -> a.grad += (1.0 / a.data) * out.grad;
        return out;
    }

    // Backpropagation
    
    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, topo, visited);

        // Zero out all gradients
        for (Value v : topo) {
            v.grad = 0.0;
        }
        
        // The gradient of output with respect to itself is 1
        this.grad = 1.0;

        // Backpropagate through the graph in reverse topological order
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i).backwardFn.run();
        }
    }

    private static void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
        if (visited.contains(v)) {
            return;
        }
        visited.add(v);
        for (Value p : v.parents) {
            buildTopo(p, topo, visited);
        }
        topo.add(v);
    }

    // Utilities
    
    public static void zeroGrad(List<Value> values) {
        if (values == null) {
            return;
        }
        for (Value v : values) {
            if (v != null) {
                v.grad = 0.0;
            }
        }
    }

    @Override
    public String toString() {
        return "Value{" +
                "data=" + data +
                ", grad=" + grad +
                ", op='" + op + '\'' +
                (label != null && !label.isEmpty() ? ", label='" + label + '\'' : "") +
                '}';
    }

}

