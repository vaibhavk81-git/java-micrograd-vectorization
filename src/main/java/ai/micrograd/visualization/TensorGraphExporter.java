package ai.micrograd.visualization;

import ai.micrograd.tensor.Tensor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * TensorGraphExporter exports computation graphs to DOT format and optionally PNG.
 * 
 * <p>Traverses the autograd graph starting from a root tensor and generates
 * a Graphviz DOT file showing nodes (tensors) and edges (operations).
 * 
 * <p>If Graphviz's 'dot' command is available on PATH, also generates a PNG image.
 * 
 * @author Vaibhav Khare
 */
public final class TensorGraphExporter {
    
    private TensorGraphExporter() {} // Prevent instantiation
    
    /**
     * Exports computation graph to DOT file and optionally PNG.
     * 
     * @param root root tensor (typically a scalar loss)
     * @param dotFile output DOT file path
     * @param pngFile output PNG file path (created only if 'dot' command exists)
     * @throws IOException if files cannot be written
     */
    public static void exportDotAndMaybePng(Tensor root, Path dotFile, Path pngFile) throws IOException {
        Files.createDirectories(dotFile.getParent());
        
        // Build graph
        Set<Tensor> visited = new HashSet<>();
        List<Tensor> nodes = new ArrayList<>();
        List<Edge> edges = new ArrayList<>();
        
        buildGraph(root, visited, nodes, edges);
        
        // Write DOT file
        writeDotFile(nodes, edges, dotFile);
        System.out.println("Saved: " + dotFile.getFileName());
        
        // Try to generate PNG with Graphviz
        if (isGraphvizAvailable()) {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                    "dot", "-Tpng", dotFile.toString(), "-o", pngFile.toString()
                );
                pb.redirectErrorStream(true);
                Process process = pb.start();
                int exitCode = process.waitFor();
                
                if (exitCode == 0) {
                    System.out.println("Saved: " + pngFile.getFileName());
                } else {
                    System.out.println("Warning: dot command failed with exit code " + exitCode);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.out.println("Warning: PNG generation interrupted");
            } catch (IOException e) {
                System.out.println("Warning: Could not generate PNG: " + e.getMessage());
            }
        } else {
            System.out.println("Note: Graphviz 'dot' not found on PATH, skipping PNG generation");
        }
    }
    
    private static void buildGraph(Tensor node, Set<Tensor> visited, List<Tensor> nodes, List<Edge> edges) {
        if (node == null || visited.contains(node)) {
            return;
        }
        
        visited.add(node);
        nodes.add(node);
        
        // Traverse parents
        if (node.getParents() != null) {
            for (Tensor parent : node.getParents()) {
                if (parent != null) {
                    edges.add(new Edge(parent, node));
                    buildGraph(parent, visited, nodes, edges);
                }
            }
        }
    }
    
    private static void writeDotFile(List<Tensor> nodes, List<Edge> edges, Path dotFile) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(dotFile)) {
            writer.write("digraph ComputationGraph {\n");
            writer.write("  rankdir=LR;\n");
            writer.write("  node [shape=box, style=rounded];\n\n");
            
            // Write nodes
            Map<Tensor, String> nodeIds = new HashMap<>();
            int idCounter = 0;
            
            for (Tensor node : nodes) {
                String nodeId = "n" + idCounter++;
                nodeIds.put(node, nodeId);
                
                String label = buildNodeLabel(node);
                writer.write(String.format("  %s [label=\"%s\"];\n", nodeId, label));
            }
            
            writer.write("\n");
            
            // Write edges
            for (Edge edge : edges) {
                String fromId = nodeIds.get(edge.from);
                String toId = nodeIds.get(edge.to);
                if (fromId != null && toId != null) {
                    writer.write(String.format("  %s -> %s;\n", fromId, toId));
                }
            }
            
            writer.write("}\n");
        }
    }
    
    private static String buildNodeLabel(Tensor node) {
        StringBuilder label = new StringBuilder();
        
        // Label
        if (node.label() != null && !node.label().isEmpty()) {
            label.append(node.label()).append("\\n");
        }
        
        // Operation
        if (node.op() != null && !node.op().isEmpty()) {
            label.append("op: ").append(node.op()).append("\\n");
        }
        
        // Shape
        label.append("shape: (").append(node.rows()).append(", ").append(node.cols()).append(")\\n");
        
        // Value (show first element if scalar or small)
        if (node.isScalar()) {
            label.append(String.format("val: %.4f\\n", node.item()));
        } else if (node.elements() <= 4) {
            double[] values = node.toArray();
            label.append("val: [");
            for (int i = 0; i < values.length; i++) {
                if (i > 0) label.append(", ");
                label.append(String.format("%.2f", values[i]));
            }
            label.append("]\\n");
        }
        
        // Gradient (show first element if scalar or small)
        if (node.requiresGrad()) {
            if (node.isScalar()) {
                double[] grads = node.gradToArray();
                label.append(String.format("grad: %.4f", grads[0]));
            } else if (node.elements() <= 4) {
                double[] grads = node.gradToArray();
                label.append("grad: [");
                for (int i = 0; i < grads.length; i++) {
                    if (i > 0) label.append(", ");
                    label.append(String.format("%.2f", grads[i]));
                }
                label.append("]");
            }
        }
        
        return label.toString();
    }
    
    private static boolean isGraphvizAvailable() {
        try {
            ProcessBuilder pb = new ProcessBuilder("dot", "-V");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }
    
    private static class Edge {
        final Tensor from;
        final Tensor to;
        
        Edge(Tensor from, Tensor to) {
            this.from = from;
            this.to = to;
        }
    }
}


