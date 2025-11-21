package ai.micrograd.visualization;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * LossPlotter generates loss curve visualizations.
 * 
 * <p>Creates a line plot of training loss over epochs using XChart.
 * All plots are generated in headless mode for server compatibility.
 * 
 * @author Vaibhav Khare
 */
public final class LossPlotter {
    
    private LossPlotter() {} // Prevent instantiation
    
    /**
     * Plots training loss curve and saves to PNG.
     * 
     * @param losses list of loss values (one per epoch)
     * @param outFile output file path
     * @throws IOException if file cannot be written
     */
    public static void plot(List<Double> losses, Path outFile) throws IOException {
        System.setProperty("java.awt.headless", "true");
        
        XYChart chart = new XYChartBuilder()
            .width(800)
            .height(600)
            .title("Training Loss")
            .xAxisTitle("Epoch")
            .yAxisTitle("Loss")
            .build();
        
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        
        List<Integer> xs = IntStream.range(0, losses.size())
                                    .boxed()
                                    .collect(Collectors.toList());
        
        chart.addSeries("loss", xs, losses);
        
        Files.createDirectories(outFile.getParent());
        BitmapEncoder.saveBitmap(chart, outFile.toString(), BitmapEncoder.BitmapFormat.PNG);
        System.out.println("Saved: " + outFile.getFileName());
    }
}


