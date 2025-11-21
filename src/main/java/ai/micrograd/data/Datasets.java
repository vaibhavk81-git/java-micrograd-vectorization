package ai.micrograd.data;

import ai.micrograd.tensor.Tensor;
import java.util.random.RandomGenerator;

/**
 * Datasets provides synthetic dataset generation for testing and demos.
 * 
 * @author Vaibhav Khare
 */
public final class Datasets {
    
    private Datasets() {} // Prevent instantiation
    
    /**
     * Container for two-moons dataset.
     * 
     * @param X feature matrix (n×2)
     * @param y labels (n×1) with values in {-1, +1}
     */
    public record MoonsData(Tensor X, Tensor y) {}
    
    /**
     * Generates the two-moons dataset for binary classification.
     * 
     * <p>Creates two interleaving half-circles (moons) with optional Gaussian noise.
     * This is a classic non-linearly separable dataset used to test neural networks.
     * 
     * <p><b>RNG Policy:</b> Uses a dedicated dataRng derived from seed to ensure
     * reproducibility independent of parameter initialization.
     * 
     * @param nSamples total number of samples (split equally between two classes)
     * @param noise standard deviation of Gaussian noise added to points
     * @param seed random seed for reproducibility
     * @return MoonsData containing features X (n×2) and labels y (n×1)
     */
    public static MoonsData makeMoons(int nSamples, double noise, long seed) {
        if (nSamples < 2) {
            throw new IllegalArgumentException("nSamples must be at least 2, got: " + nSamples);
        }
        if (noise < 0) {
            throw new IllegalArgumentException("noise must be non-negative, got: " + noise);
        }
        
        // Use dataRng with XOR pattern to separate from initRng
        RandomGenerator dataRng = java.util.random.RandomGeneratorFactory
            .of("L64X128MixRandom")
            .create(seed ^ 0x9E3779B97F4A7C15L);
        
        int nOuter = nSamples / 2;
        int nInner = nSamples - nOuter;
        
        Tensor X = new Tensor(nSamples, 2, false);
        Tensor y = new Tensor(nSamples, 1, false);
        
        // Generate outer moon (+1)
        // Guard against division by zero for small sample counts
        for (int i = 0; i < nOuter; i++) {
            double angle = Math.PI * i / Math.max(nOuter - 1, 1);
            double x =  Math.cos(angle);
            double yy = Math.sin(angle);

            x += noise * gaussianNoise(dataRng);
            yy += noise * gaussianNoise(dataRng);

            X.set(i, 0, x);
            X.set(i, 1, yy);
            y.set(i, 0, 1.0);
        }

        // Generate inner moon (-1)
        // Guard against division by zero for small sample counts
        for (int i = 0; i < nInner; i++) {
            double angle = Math.PI * i / Math.max(nInner - 1, 1);
            double x = 1.0 - Math.cos(angle);     // horizontal shift
            double yy = -Math.sin(angle) + 0.5;   // vertical flip + shift

            x += noise * gaussianNoise(dataRng);
            yy += noise * gaussianNoise(dataRng);

            X.set(nOuter + i, 0, x);
            X.set(nOuter + i, 1, yy);
            y.set(nOuter + i, 0, -1.0);
        }

        
        return new MoonsData(X, y);
    }
    
    /**
     * Generates a sample from standard normal distribution using Box-Muller transform.
     */
    private static double gaussianNoise(RandomGenerator rng) {
        double u1 = rng.nextDouble();
        double u2 = rng.nextDouble();
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }
}

