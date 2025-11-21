# Java Micrograd Vectorization

A vectorized tensor autograd engine with 2-D tensor support, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and the [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) series.

This project extends a scalar autograd engine with efficient vectorized operations, mini-batch training, and a complete two-moons binary classifier demo.

## Features

### Core Engine
- **Scalar Autograd**: Original `Value`-based computation graph (from foundation project)
- **Vectorized Tensors**: 2-D tensors with device-aware storage abstraction
- **Multi-Precision Support**: FP64 (double) and FP32 (float) with full gradient support
- **Backend Abstraction**: CPU backend with clean extension points for future GPU support
- **Automatic Differentiation**: Full backpropagation with gradient accumulation
- **Deterministic Training**: Reproducible experiments with separate RNGs for init and data

### Tensor Operations
- **Element-wise**: add, mul, tanh, relu
- **Broadcast**: addRowVector (bias addition)
- **Matrix Operations**: matmul with efficient gradients
- **Reductions**: sum/mean along axis (0 or 1)

### Neural Network Components
- **Linear Layer**: Fully-connected layer with Xavier uniform initialization
- **VectorMLP**: Multi-layer perceptron wrapper with configurable activation
- **Activations**: tanh, ReLU
- **Loss Functions**: Hinge loss with L2 regularization

### Training Infrastructure
- **SGD Optimizer**: Stochastic gradient descent with configurable learning rate
- **Mini-batch Training**: Efficient batched forward/backward passes
- **Deterministic Shuffling**: Reproducible data ordering per epoch

### Enhancements
- **Precision Control**: FP64 and FP32 fully supported on CPU
- **Device Management**: CPU-only today, with environment variable hooks for future GPU
- **Profiler**: Forward/backward/step timing utilities
- **Visualization**: Loss curves and decision boundary plots with XChart

## API Overview

### Tensor Creation

```java
// Default precision (FP64) and device (CPU)
Tensor t1 = new Tensor(2, 3, false);

// Explicit precision
Tensor t2 = new Tensor(2, 3, Precision.FP32, false);

// Factory methods
Tensor zeros = Tensor.zeros(3, 4, true);
Tensor ones = Tensor.ones(2, 2, Precision.FP32, false);
Tensor rand = Tensor.rand(5, 5, rng, -1.0, 1.0, true);

// From arrays
Tensor t3 = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}}, true);
Tensor t4 = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}}, Precision.FP32, true);

// Template-based creation (preserves precision and device)
Tensor zerosLike = Tensor.zerosLike(t1, false);
Tensor onesLike = Tensor.onesLike(t1, false);
Tensor randLike = Tensor.randLike(t1, rng, 0.0, 1.0, false);
```

### Precision Control

```java
// Set global default precision
Tensor.setDefaultPrecision(Precision.FP32);

// Query precision
Precision p = tensor.precision();  // FP64 or FP32

// All operations preserve precision
Tensor a = Tensor.ones(2, 2, Precision.FP32, false);
Tensor b = Tensor.ones(2, 2, Precision.FP32, false);
Tensor c = TensorOps.add(a, b);  // c is also FP32
```

### Device Management

```java
// Query available devices
DeviceManager dm = DeviceManager.get();
List<DeviceType> devices = dm.availableDevices();  // [CPU]
DeviceType defaultDev = dm.defaultDevice();  // CPU

// Check for GPU warnings (future-proofing)
Optional<String> warning = dm.gpuRequestWarning();
warning.ifPresent(System.out::println);

// Environment variables (for future GPU support)
// MICROGRAD_DEVICE=cpu
// MICROGRAD_ENABLE_GPU=1
```

### Tensor Operations

```java
// Element-wise
Tensor c = TensorOps.add(a, b);
Tensor d = TensorOps.mul(a, b);
Tensor e = TensorOps.tanh(a);
Tensor f = TensorOps.relu(a);

// Matrix multiplication
Tensor g = TensorOps.matmul(a, b);

// Broadcast
Tensor h = TensorOps.addRowVector(matrix, rowVec);

// Reductions
Tensor i = TensorOps.sum(a, 0);  // sum over rows
Tensor j = TensorOps.mean(a, 1);  // mean over columns
```

### Autograd

```java
Tensor x = Tensor.fromArray(new double[][]{{1, 2}}, true);
Tensor y = TensorOps.mul(x, x);
Tensor loss = TensorOps.sum(y, 1);

Tensor.backward(loss);

double[] grad = x.gradToArray();  // [2.0, 4.0]
```

## Quick Start

### Prerequisites
- Java 21
- Gradle (wrapper included)

### Build
```bash
./gradlew build
```

### Run Tests
```bash
./gradlew test
```

All 34 tests should pass, validating:
- Tensor shape checking
- Gradient correctness (central difference)
- Component functionality (SGD, MLP, Profiler, Precision)
- End-to-end training pipeline

### Train Two-Moons Classifier
```bash
./gradlew run --args="--hidden 4 --epochs 200 --lr 0.1 --batch 32 --seed 42 --noise 0.2 --samples 400"
```

Output:
- `runs/<timestamp>/loss.png`: Training loss curve
- `runs/<timestamp>/decision_boundary.png`: Model decision regions
- `runs/<timestamp>/metrics.json`: Final accuracies and hyperparameters

### CLI Options

```
--hidden <int>       Hidden layer size (default: 4)
--epochs <int>       Number of epochs, max 200 (default: 200)
--lr <double>        Learning rate (default: 0.1)
--batch <int>        Batch size (default: 32)
--seed <long>        Random seed (default: 42)
--l2 <double>        L2 regularization strength (default: 0.0)
--samples <int>      Number of samples (default: 400)
--noise <double>     Dataset noise level (default: 0.2)
--outDir <path>      Output directory (default: runs/<timestamp>/)
--precision <FP64|FP32>  Precision mode (default: FP64)
--device <cpu|gpu>   Device selection (default: cpu, GPU not yet implemented)
--profile            Enable profiling (default: false)
```

### Example with Profiling
```bash
./gradlew run --args="--hidden 8 --epochs 100 --lr 0.05 --profile"
```

Outputs timing breakdown:
```
=== Profiling Results ===
Total times:
  forward: 45.23 ms
  backward: 78.91 ms
  step: 12.34 ms
```

## Architecture

### Package Structure
```
ai.micrograd/
├── autograd/          # Scalar autograd engine (Value)
├── nn/                # Neural network components
│   ├── Module.java    # Base interface
│   ├── Neuron.java    # Scalar neuron
│   ├── Layer.java     # Scalar layer
│   ├── MLP.java       # Scalar MLP
│   ├── Linear.java    # Vectorized linear layer
│   ├── VectorMLP.java # Vectorized MLP
│   ├── Activations.java
│   └── Losses.java
├── tensor/            # Vectorized tensor engine
│   ├── Tensor.java
│   ├── TensorOps.java
│   ├── TensorBackprop.java
│   └── Precision.java
├── optim/             # Optimizers
│   └── SGD.java
├── data/              # Dataset generation
│   └── Datasets.java
├── util/              # Utilities
│   └── Profiler.java
└── demo/              # Demo applications
    └── MoonsBinaryClassifier.java
```

### Tensor Storage
Tensors use **flat `double[]` arrays** in row-major order:
```java
// For tensor with shape (m, n):
// Element at (row, col) is at index: row * n + col
```

This provides:
- Cache-friendly memory layout
- Efficient SIMD potential
- Simple indexing arithmetic

### Gradient Computation
All operations use **gradient accumulation** (`+=`) to support:
- Multivariate chain rule
- Shared parameters
- Multiple gradient paths

Example:
```java
Tensor a = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}}, true);
Tensor b = TensorOps.add(a, a);  // a used twice
Tensor loss = TensorOps.sum(b, 0);
loss = TensorOps.sum(loss, 1);
Tensor.backward(loss);
// a.grad accumulates gradients from both paths
```

### RNG Policy
Two separate random generators ensure reproducibility:
```java
initRng = RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
dataRng = RandomGeneratorFactory.of("L64X128MixRandom").create(seed ^ 0x9E3779B97F4A7C15L);
```

- `initRng`: Parameter initialization (weights, biases)
- `dataRng`: Data shuffling, noise generation

## Scalar vs Vectorized Paths

### Scalar Path (Original)
```java
// Scalar operations on individual values
Value x = new Value(2.0, "x");
Value y = new Value(3.0, "y");
Value z = x.mul(y).add(x);
z.backward();
```

### Vectorized Path (New)
```java
// Batched operations on tensors
Tensor X = Tensor.fromArray(new double[][]{{1, 2}, {3, 4}}, true);
Tensor W = Tensor.rand(2, 3, rng, -1, 1, true);
Tensor Y = TensorOps.matmul(X, W);
Tensor loss = TensorOps.mean(Y, 0);
loss = TensorOps.sum(loss, 1);
Tensor.backward(loss);
```

Both paths coexist without interference.

## Implementation Notes

### Gradient Checking
All operations are validated with central difference approximation:
```java
numericalGrad = (f(x + ε) - f(x - ε)) / (2ε)
relError = |numerical - analytical| / (|numerical| + |analytical| + 1e-8)
```

Tests ensure `relError < 1e-4` for ε = 1e-5.

### Known Limitations
1. **2-D Only**: No support for higher-dimensional tensors
2. **CPU Only**: No GPU acceleration (architecture is GPU-ready)
3. **Test Accuracy**: Pipeline test has relaxed threshold due to overfitting on small dataset

### Future Enhancements
- GPU acceleration (CUDA/OpenCL) - backend abstraction is ready
- Additional optimizers (Adam, RMSprop)
- Convolutional layers
- Higher-dimensional tensors
- Mixed-precision training

## Testing

### Test Coverage
- **TensorShapeTest**: Shape validation for all operations
- **TensorGradCheckTest**: Numerical gradient verification
- **MoonsPipelineTest**: End-to-end training validation
- **ComponentTests**: SGD, VectorMLP, Precision, Profiler
- **PrecisionTest**: FP32/FP64 operations and gradient correctness

### Run Specific Tests
```bash
./gradlew test --tests TensorShapeTest
./gradlew test --tests TensorGradCheckTest
./gradlew test --tests MoonsPipelineTest
./gradlew test --tests ComponentTests
```

### Coverage Report
```bash
./gradlew jacocoTestReport
open build/reports/jacoco/test/html/index.html
```

## Dependencies

- **picocli 4.7.5**: CLI argument parsing
- **xchart 3.8.6**: Plotting and visualization
- **JUnit Jupiter 5.10.2**: Testing framework
- **SLF4J Simple 2.0.13**: Logging (optional)

## Reproducibility

All experiments are fully reproducible with fixed seeds:
```bash
# Same command always produces same results
./gradlew run --args="--seed 42 --epochs 100"
```

The RNG policy ensures:
- Same parameter initialization
- Same data shuffling order
- Same noise samples

## Performance

Typical training time (M1 Mac, Java 21):
- 200 epochs, 400 samples, batch=32: ~0.5 seconds
- Profiling overhead: ~10-15%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is for educational purposes, inspired by Andrej Karpathy's micrograd.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for micrograd and nn-zero-to-hero
- [PyTorch](https://pytorch.org/) for API design inspiration
- Foundation project contributors for the original scalar autograd implementation

## Contact

For questions or issues, please refer to the project repository.

