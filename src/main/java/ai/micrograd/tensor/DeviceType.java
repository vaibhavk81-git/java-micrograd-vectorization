package ai.micrograd.tensor;

/**
 * Supported tensor device targets.
 *
 * <p>Only {@link #CPU} is implemented today, but the enum intentionally
 * reserves {@link #GPU} so higher-level APIs can start accepting device hints
 * without having to change their signatures when GPU support arrives.</p>
 */
public enum DeviceType {
    CPU("cpu"),
    GPU("gpu");

    private final String cliValue;

    DeviceType(String cliValue) {
        this.cliValue = cliValue;
    }

    /**
     * Returns the lowercase identifier used by CLI/config layers.
     */
    public String cliValue() {
        return cliValue;
    }

    /**
     * Parses a device flag such as "cpu" or "gpu".
     *
     * @param token user supplied string (case insensitive)
     * @return matching device type
     * @throws IllegalArgumentException if the token does not match a device
     */
    public static DeviceType fromString(String token) {
        if (token == null) {
            throw new IllegalArgumentException("Device token must not be null");
        }
        String normalized = token.trim().toLowerCase();
        for (DeviceType type : values()) {
            if (type.cliValue.equals(normalized)) {
                return type;
            }
        }
        throw new IllegalArgumentException(
            "Unknown device type: " + token + ". Supported: cpu");
    }

    public boolean isCpu() {
        return this == CPU;
    }

    public boolean isGpu() {
        return this == GPU;
    }
}


