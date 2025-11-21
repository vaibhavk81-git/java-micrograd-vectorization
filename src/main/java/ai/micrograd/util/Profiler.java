package ai.micrograd.util;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Profiler provides simple timing utilities for performance measurement.
 * 
 * <p>Useful for understanding where time is spent during training:
 * forward pass, backward pass, or parameter updates.
 * 
 * @author Vaibhav Khare
 */
public final class Profiler {
    
    private Profiler() {} // Prevent instantiation
    
    /**
     * Times the execution of a runnable and returns elapsed nanoseconds.
     * 
     * @param r runnable to time
     * @return elapsed time in nanoseconds
     */
    public static long timeNanos(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        return System.nanoTime() - t0;
    }
    
    /**
     * Profiles a training step by timing forward, backward, and update phases.
     * 
     * <p>Returns a map with keys: "forward", "backward", "step"
     * and values in nanoseconds.
     * 
     * @param forward forward pass runnable
     * @param backward backward pass runnable
     * @param step parameter update runnable
     * @return map of phase names to elapsed nanoseconds
     */
    public static Map<String, Long> profileTrainStep(Runnable forward, Runnable backward, Runnable step) {
        Map<String, Long> times = new LinkedHashMap<>();
        times.put("forward", timeNanos(forward));
        times.put("backward", timeNanos(backward));
        times.put("step", timeNanos(step));
        return times;
    }
    
    /**
     * Converts nanoseconds to milliseconds.
     * 
     * @param nanos time in nanoseconds
     * @return time in milliseconds
     */
    public static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }
    
    /**
     * Formats timing results as a human-readable string.
     * 
     * @param times map of phase names to nanoseconds
     * @return formatted string
     */
    public static String formatTimes(Map<String, Long> times) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Long> entry : times.entrySet()) {
            if (sb.length() > 0) sb.append(", ");
            sb.append(entry.getKey()).append("=");
            sb.append(String.format("%.2fms", nanosToMillis(entry.getValue())));
        }
        return sb.toString();
    }
}

