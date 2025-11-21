package ai.micrograd.visualization;

import ai.micrograd.tensor.Tensor;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * DecisionBoundaryPlotter generates decision boundary visualizations.
 * 
 * <p>Creates a scatter plot showing:
 * <ul>
 *   <li>Background grid colored by predicted class (300×300 resolution)</li>
 *   <li>Training data points overlaid with distinct markers</li>
 * </ul>
 * 
 * <p>All plots are generated in headless mode for server compatibility.
 * 
 * @author Vaibhav Khare
 */
public final class DecisionBoundaryPlotter {
    
    private DecisionBoundaryPlotter() {} // Prevent instantiation
    
    /**
     * Plots decision boundary and saves to PNG.
     * 
     * @param model trained model implementing TrainedModel interface
     * @param xTrain training data features (n×2 array)
     * @param yTrain training data labels (n-length array, values in {-1, +1})
     * @param outFile output file path
     * @throws IOException if file cannot be written
     */
    public static void plot(TrainedModel model, double[][] xTrain, double[] yTrain, Path outFile) 
            throws IOException {
        System.setProperty("java.awt.headless", "true");
        
        // Find data bounds
        double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
        double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
        
        for (double[] point : xTrain) {
            xMin = Math.min(xMin, point[0]);
            xMax = Math.max(xMax, point[0]);
            yMin = Math.min(yMin, point[1]);
            yMax = Math.max(yMax, point[1]);
        }
        
        // Add margin
        double margin = 0.5;
        xMin -= margin;
        xMax += margin;
        yMin -= margin;
        yMax += margin;
        
        // Create 300×300 grid
        int gridSize = 300;
        double xStep = (xMax - xMin) / gridSize;
        double yStep = (yMax - yMin) / gridSize;
        
        List<Double> class1X = new ArrayList<>();
        List<Double> class1Y = new ArrayList<>();
        List<Double> class2X = new ArrayList<>();
        List<Double> class2Y = new ArrayList<>();
        
        // Evaluate model on grid
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                double x = xMin + i * xStep;
                double y = yMin + j * yStep;
                
                double score = model.score(x, y);
                
                if (score >= 0) {
                    class1X.add(x);
                    class1Y.add(y);
                } else {
                    class2X.add(x);
                    class2Y.add(y);
                }
            }
        }
        
        // Create chart
        XYChart chart = new XYChartBuilder()
            .width(800)
            .height(600)
            .title("Decision Boundary")
            .xAxisTitle("Feature 1")
            .yAxisTitle("Feature 2")
            .build();
        
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setMarkerSize(3);
        
        // Add background regions
        if (!class1X.isEmpty()) {
            XYSeries series1 = chart.addSeries("Region +1", class1X, class1Y);
            series1.setMarker(SeriesMarkers.CIRCLE);
        }
        if (!class2X.isEmpty()) {
            XYSeries series2 = chart.addSeries("Region -1", class2X, class2Y);
            series2.setMarker(SeriesMarkers.CIRCLE);
        }
        
        // Add training data points
        List<Double> posX = new ArrayList<>();
        List<Double> posY = new ArrayList<>();
        List<Double> negX = new ArrayList<>();
        List<Double> negY = new ArrayList<>();
        
        for (int i = 0; i < xTrain.length; i++) {
            if (yTrain[i] > 0) {
                posX.add(xTrain[i][0]);
                posY.add(xTrain[i][1]);
            } else {
                negX.add(xTrain[i][0]);
                negY.add(xTrain[i][1]);
            }
        }
        
        if (!posX.isEmpty()) {
            XYSeries series = chart.addSeries("Data +1", posX, posY);
            series.setMarker(SeriesMarkers.DIAMOND);
        }
        if (!negX.isEmpty()) {
            XYSeries series = chart.addSeries("Data -1", negX, negY);
            series.setMarker(SeriesMarkers.DIAMOND);
        }
        
        Files.createDirectories(outFile.getParent());
        BitmapEncoder.saveBitmap(chart, outFile.toString(), BitmapEncoder.BitmapFormat.PNG);
        System.out.println("Saved: " + outFile.getFileName());
    }
    
    /**
     * Convenience overload accepting Tensor inputs.
     * 
     * @param model trained model
     * @param X training data features (Tensor, n×2)
     * @param y training data labels (Tensor, n×1, values in {-1, +1})
     * @param outFile output file path
     * @throws IOException if file cannot be written
     */
    public static void plot(TrainedModel model, Tensor X, Tensor y, Path outFile) throws IOException {
        // Convert tensors to arrays
        double[][] xArray = new double[X.rows()][X.cols()];
        double[] yArray = new double[y.rows()];
        
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                xArray[i][j] = X.get(i, j);
            }
            yArray[i] = y.get(i, 0);
        }
        
        plot(model, xArray, yArray, outFile);
    }
}


