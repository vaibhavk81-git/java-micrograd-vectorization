package ai.micrograd.tensor;

import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Central place to reason about device availability. Today it only exposes CPU,
 * but it watches a couple of environment variables so that we can add GPU
 * support without changing higher layers again.
 */
public final class DeviceManager {

    private static final String ENV_PREFERRED_DEVICE = "MICROGRAD_DEVICE";
    private static final String ENV_ENABLE_GPU = "MICROGRAD_ENABLE_GPU";
    private static final DeviceManager INSTANCE = new DeviceManager();

    private final List<DeviceType> availableDevices;
    private final DeviceType defaultDevice;
    private final boolean gpuFlagDetected;
    private final String preferredToken;

    private DeviceManager() {
        this.availableDevices = Collections.singletonList(DeviceType.CPU);
        Map<String, String> env = System.getenv();
        this.preferredToken = env.getOrDefault(ENV_PREFERRED_DEVICE, "").trim();
        this.gpuFlagDetected = env.containsKey(ENV_ENABLE_GPU);
        this.defaultDevice = resolveDefaultDevice();
    }

    public static DeviceManager get() {
        return INSTANCE;
    }

    public List<DeviceType> availableDevices() {
        return availableDevices;
    }

    public DeviceType defaultDevice() {
        return defaultDevice;
    }

    public boolean isAvailable(DeviceType type) {
        return availableDevices.contains(type);
    }

    public Optional<String> gpuRequestWarning() {
        if (gpuFlagDetected) {
            return Optional.of("GPU flag detected via " + ENV_ENABLE_GPU +
                " but GPU backend is not implemented yet. Falling back to CPU.");
        }
        if (!preferredToken.isEmpty() && parseDeviceToken(preferredToken) == DeviceType.GPU) {
            return Optional.of("Preferred device '" + preferredToken +
                "' is GPU, but only CPU is available at the moment.");
        }
        return Optional.empty();
    }

    public Optional<DeviceType> preferredDeviceFromEnv() {
        DeviceType parsed = parseDeviceToken(preferredToken);
        return Optional.ofNullable(parsed);
    }

    public DeviceType resolve(DeviceType requested) {
        if (requested == null) {
            return defaultDevice;
        }
        return isAvailable(requested) ? requested : defaultDevice;
    }

    private DeviceType resolveDefaultDevice() {
        DeviceType envPreferred = parseDeviceToken(preferredToken);
        if (envPreferred != null && isAvailable(envPreferred)) {
            return envPreferred;
        }
        return DeviceType.CPU;
    }

    private DeviceType parseDeviceToken(String token) {
        if (token == null || token.isBlank()) {
            return null;
        }
        String normalized = token.trim().toLowerCase(Locale.ENGLISH);
        for (DeviceType type : DeviceType.values()) {
            if (type.cliValue().equals(normalized)) {
                return type;
            }
        }
        return null;
    }
}

