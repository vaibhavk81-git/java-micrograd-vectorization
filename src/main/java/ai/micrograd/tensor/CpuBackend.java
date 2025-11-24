package ai.micrograd.tensor;

/**
 * CPU backend that executes tensor math using plain Java loops.
 *
 * <p>The implementation supports both FP64 and FP32 precisions and keeps
 * gradients in the same precision as the forward buffers. Extending to new
 * devices involves providing another backend implementation and registering it
 * inside {@link TensorBackendRegistry}.</p>
 */
final class CpuBackend implements TensorBackend {

    @Override
    public DeviceType deviceType() {
        return DeviceType.CPU;
    }

    @Override
    public TensorStorage allocate(int size, Precision precision, boolean requiresGrad) {
        return new CpuTensorStorage(size, precision, requiresGrad);
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        ensureBinaryCompat("add", a, b);
        Tensor out = Tensor.emptyLike(a, a.requiresGrad() || b.requiresGrad());
        CpuTensorStorage outStore = storageOf(out);
        if (a.precision() == Precision.FP64) {
            double[] aData = storageOf(a).data64();
            double[] bData = storageOf(b).data64();
            double[] outData = outStore.data64();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = aData[i] + bData[i];
            }
        } else {
            float[] aData = storageOf(a).data32();
            float[] bData = storageOf(b).data32();
            float[] outData = outStore.data32();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = (float) (aData[i] + bData[i]);
            }
        }

        if (out.requiresGrad()) {
            out.setParents(a, b);
            out.setOp("add");
            out.setBackwardFn(() -> {
                if (a.requiresGrad()) {
                    if (a.precision() == Precision.FP64) {
                        double[] aGrad = storageOf(a).grad64();
                        double[] outGrad = outStore.grad64();
                        accumulate(aGrad, outGrad);
                    } else {
                        float[] aGrad = storageOf(a).grad32();
                        float[] outGrad = outStore.grad32();
                        accumulate(aGrad, outGrad);
                    }
                }
                if (b.requiresGrad()) {
                    if (b.precision() == Precision.FP64) {
                        double[] bGrad = storageOf(b).grad64();
                        double[] outGrad = outStore.grad64();
                        accumulate(bGrad, outGrad);
                    } else {
                        float[] bGrad = storageOf(b).grad32();
                        float[] outGrad = outStore.grad32();
                        accumulate(bGrad, outGrad);
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        ensureBinaryCompat("mul", a, b);
        Tensor out = Tensor.emptyLike(a, a.requiresGrad() || b.requiresGrad());
        CpuTensorStorage outStore = storageOf(out);
        if (a.precision() == Precision.FP64) {
            double[] aData = storageOf(a).data64();
            double[] bData = storageOf(b).data64();
            double[] outData = outStore.data64();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = aData[i] * bData[i];
            }
        } else {
            float[] aData = storageOf(a).data32();
            float[] bData = storageOf(b).data32();
            float[] outData = outStore.data32();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = (float) (aData[i] * bData[i]);
            }
        }

        if (out.requiresGrad()) {
            out.setParents(a, b);
            out.setOp("mul");
            out.setBackwardFn(() -> {
                if (a.requiresGrad()) {
                    if (a.precision() == Precision.FP64) {
                        double[] aGrad = storageOf(a).grad64();
                        double[] bData = storageOf(b).data64();
                        double[] outGrad = outStore.grad64();
                        for (int i = 0; i < outGrad.length; i++) {
                            aGrad[i] += bData[i] * outGrad[i];
                        }
                    } else {
                        float[] aGrad = storageOf(a).grad32();
                        float[] bData = storageOf(b).data32();
                        float[] outGrad = outStore.grad32();
                        for (int i = 0; i < outGrad.length; i++) {
                            aGrad[i] += bData[i] * outGrad[i];
                        }
                    }
                }
                if (b.requiresGrad()) {
                    if (b.precision() == Precision.FP64) {
                        double[] bGrad = storageOf(b).grad64();
                        double[] aData = storageOf(a).data64();
                        double[] outGrad = outStore.grad64();
                        for (int i = 0; i < outGrad.length; i++) {
                            bGrad[i] += aData[i] * outGrad[i];
                        }
                    } else {
                        float[] bGrad = storageOf(b).grad32();
                        float[] aData = storageOf(a).data32();
                        float[] outGrad = outStore.grad32();
                        for (int i = 0; i < outGrad.length; i++) {
                            bGrad[i] += aData[i] * outGrad[i];
                        }
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor tanh(Tensor x) {
        Tensor out = Tensor.emptyLike(x, x.requiresGrad());
        CpuTensorStorage outStore = storageOf(out);
        if (x.precision() == Precision.FP64) {
            double[] outData = outStore.data64();
            double[] xData = storageOf(x).data64();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = Math.tanh(xData[i]);
            }
        } else {
            float[] outData = outStore.data32();
            float[] xData = storageOf(x).data32();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = (float) Math.tanh(xData[i]);
            }
        }

        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("tanh");
            out.setBackwardFn(() -> {
                if (x.precision() == Precision.FP64) {
                    double[] xGrad = storageOf(x).grad64();
                    double[] outData = outStore.data64();
                    double[] outGrad = outStore.grad64();
                    for (int i = 0; i < outGrad.length; i++) {
                        double t = outData[i];
                        xGrad[i] += (1.0 - t * t) * outGrad[i];
                    }
                } else {
                    float[] xGrad = storageOf(x).grad32();
                    float[] outData = outStore.data32();
                    float[] outGrad = outStore.grad32();
                    for (int i = 0; i < outGrad.length; i++) {
                        float t = outData[i];
                        xGrad[i] += (1.0f - t * t) * outGrad[i];
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor relu(Tensor x) {
        Tensor out = Tensor.emptyLike(x, x.requiresGrad());
        CpuTensorStorage outStore = storageOf(out);
        if (x.precision() == Precision.FP64) {
            double[] outData = outStore.data64();
            double[] xData = storageOf(x).data64();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = Math.max(0.0, xData[i]);
            }
        } else {
            float[] outData = outStore.data32();
            float[] xData = storageOf(x).data32();
            for (int i = 0; i < outData.length; i++) {
                outData[i] = Math.max(0.0f, xData[i]);
            }
        }

        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("relu");
            out.setBackwardFn(() -> {
                if (x.precision() == Precision.FP64) {
                    double[] xGrad = storageOf(x).grad64();
                    double[] xData = storageOf(x).data64();
                    double[] outGrad = outStore.grad64();
                    for (int i = 0; i < outGrad.length; i++) {
                        if (xData[i] > 0) {
                            xGrad[i] += outGrad[i];
                        }
                    }
                } else {
                    float[] xGrad = storageOf(x).grad32();
                    float[] xData = storageOf(x).data32();
                    float[] outGrad = outStore.grad32();
                    for (int i = 0; i < outGrad.length; i++) {
                        if (xData[i] > 0) {
                            xGrad[i] += outGrad[i];
                        }
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor addRowVector(Tensor matrix, Tensor rowVec) {
        if (rowVec.rows() != 1) {
            throw new IllegalArgumentException(
                "rowVec must have 1 row, got: " + rowVec.rows());
        }
        if (matrix.cols() != rowVec.cols()) {
            throw new IllegalArgumentException(
                "Column mismatch: matrix has " + matrix.cols() + " cols, row vec has " + rowVec.cols());
        }
        ensureSameDeviceAndPrecision("addRowVector", matrix, rowVec);

        Tensor out = new Tensor(matrix.rows(), matrix.cols(), matrix.precision(),
            matrix.device(), matrix.requiresGrad() || rowVec.requiresGrad());
        CpuTensorStorage outStore = storageOf(out);
        if (matrix.precision() == Precision.FP64) {
            double[] matrixData = storageOf(matrix).data64();
            double[] rowData = storageOf(rowVec).data64();
            double[] outData = outStore.data64();
            for (int r = 0; r < matrix.rows(); r++) {
                for (int c = 0; c < matrix.cols(); c++) {
                    int idx = r * matrix.cols() + c;
                    outData[idx] = matrixData[idx] + rowData[c];
                }
            }
        } else {
            float[] matrixData = storageOf(matrix).data32();
            float[] rowData = storageOf(rowVec).data32();
            float[] outData = outStore.data32();
            for (int r = 0; r < matrix.rows(); r++) {
                for (int c = 0; c < matrix.cols(); c++) {
                    int idx = r * matrix.cols() + c;
                    outData[idx] = (float) (matrixData[idx] + rowData[c]);
                }
            }
        }

        if (out.requiresGrad()) {
            out.setParents(matrix, rowVec);
            out.setOp("addRowVector");
            out.setBackwardFn(() -> {
                if (matrix.requiresGrad()) {
                    if (matrix.precision() == Precision.FP64) {
                        double[] matrixGrad = storageOf(matrix).grad64();
                        double[] outGrad = outStore.grad64();
                        accumulate(matrixGrad, outGrad);
                    } else {
                        float[] matrixGrad = storageOf(matrix).grad32();
                        float[] outGrad = outStore.grad32();
                        accumulate(matrixGrad, outGrad);
                    }
                }
                if (rowVec.requiresGrad()) {
                    if (rowVec.precision() == Precision.FP64) {
                        double[] rowGrad = storageOf(rowVec).grad64();
                        double[] outGrad = outStore.grad64();
                        for (int c = 0; c < rowVec.cols(); c++) {
                            double sum = 0.0;
                            for (int r = 0; r < out.rows(); r++) {
                                sum += outGrad[r * out.cols() + c];
                            }
                            rowGrad[c] += sum;
                        }
                    } else {
                        float[] rowGrad = storageOf(rowVec).grad32();
                        float[] outGrad = outStore.grad32();
                        for (int c = 0; c < rowVec.cols(); c++) {
                            float sum = 0.0f;
                            for (int r = 0; r < out.rows(); r++) {
                                sum += outGrad[r * out.cols() + c];
                            }
                            rowGrad[c] += sum;
                        }
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor sum(Tensor x, int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }

        final Tensor out;
        if (axis == 0) {
            out = new Tensor(1, x.cols(), x.precision(), x.device(), x.requiresGrad());
        } else {
            out = new Tensor(x.rows(), 1, x.precision(), x.device(), x.requiresGrad());
        }
        CpuTensorStorage outStore = storageOf(out);

        if (x.precision() == Precision.FP64) {
            double[] xData = storageOf(x).data64();
            double[] outData = outStore.data64();
            if (axis == 0) {
                for (int c = 0; c < x.cols(); c++) {
                    double sum = 0.0;
                    for (int r = 0; r < x.rows(); r++) {
                        sum += xData[r * x.cols() + c];
                    }
                    outData[c] = sum;
                }
            } else {
                for (int r = 0; r < x.rows(); r++) {
                    double sum = 0.0;
                    for (int c = 0; c < x.cols(); c++) {
                        sum += xData[r * x.cols() + c];
                    }
                    outData[r] = sum;
                }
            }
        } else {
            float[] xData = storageOf(x).data32();
            float[] outData = outStore.data32();
            if (axis == 0) {
                for (int c = 0; c < x.cols(); c++) {
                    float sum = 0.0f;
                    for (int r = 0; r < x.rows(); r++) {
                        sum += xData[r * x.cols() + c];
                    }
                    outData[c] = sum;
                }
            } else {
                for (int r = 0; r < x.rows(); r++) {
                    float sum = 0.0f;
                    for (int c = 0; c < x.cols(); c++) {
                        sum += xData[r * x.cols() + c];
                    }
                    outData[r] = sum;
                }
            }
        }

        if (x.requiresGrad()) {
            out.setParents(x);
            out.setOp("sum(axis=" + axis + ")");
            out.setBackwardFn(() -> broadcastReduceGrad(x, out, axis, 1.0));
        }
        return out;
    }

    @Override
    public Tensor mean(Tensor x, int axis) {
        Tensor sum = sum(x, axis);
        int count = (axis == 0) ? x.rows() : x.cols();
        CpuTensorStorage sumStore = storageOf(sum);
        if (x.precision() == Precision.FP64) {
            double[] data = sumStore.data64();
            for (int i = 0; i < data.length; i++) {
                data[i] /= count;
            }
        } else {
            float[] data = sumStore.data32();
            for (int i = 0; i < data.length; i++) {
                data[i] /= count;
            }
        }

        if (x.requiresGrad()) {
            sum.setBackwardFn(() -> broadcastReduceGrad(x, sum, axis, 1.0 / count));
        }
        return sum;
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (a.cols() != b.rows()) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch for matmul: (%d, %d) @ (%d, %d)",
                    a.rows(), a.cols(), b.rows(), b.cols()));
        }
        ensureSameDeviceAndPrecision("matmul", a, b);

        int m = a.rows();
        int k = a.cols();
        int n = b.cols();
        boolean requiresGrad = a.requiresGrad() || b.requiresGrad();

        Tensor out = new Tensor(m, n, a.precision(), a.device(), requiresGrad);
        CpuTensorStorage outStore = storageOf(out);

        if (a.precision() == Precision.FP64) {
            double[] aData = storageOf(a).data64();
            double[] bData = storageOf(b).data64();
            double[] outData = outStore.data64();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int kk = 0; kk < k; kk++) {
                        sum += aData[i * k + kk] * bData[kk * n + j];
                    }
                    outData[i * n + j] = sum;
                }
            }
        } else {
            float[] aData = storageOf(a).data32();
            float[] bData = storageOf(b).data32();
            float[] outData = outStore.data32();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < k; kk++) {
                        sum += aData[i * k + kk] * bData[kk * n + j];
                    }
                    outData[i * n + j] = sum;
                }
            }
        }

        if (requiresGrad) {
            out.setParents(a, b);
            out.setOp("matmul");
            out.setBackwardFn(() -> {
                if (a.requiresGrad()) {
                    if (a.precision() == Precision.FP64) {
                        double[] aGrad = storageOf(a).grad64();
                        double[] outGrad = outStore.grad64();
                        double[] bData = storageOf(b).data64();
                        for (int i = 0; i < m; i++) {
                            for (int kk = 0; kk < k; kk++) {
                                double sum = 0.0;
                                for (int j = 0; j < n; j++) {
                                    sum += outGrad[i * n + j] * bData[kk * n + j];
                                }
                                aGrad[i * k + kk] += sum;
                            }
                        }
                    } else {
                        float[] aGrad = storageOf(a).grad32();
                        float[] outGrad = outStore.grad32();
                        float[] bData = storageOf(b).data32();
                        for (int i = 0; i < m; i++) {
                            for (int kk = 0; kk < k; kk++) {
                                float sum = 0.0f;
                                for (int j = 0; j < n; j++) {
                                    sum += outGrad[i * n + j] * bData[kk * n + j];
                                }
                                aGrad[i * k + kk] += sum;
                            }
                        }
                    }
                }
                if (b.requiresGrad()) {
                    if (b.precision() == Precision.FP64) {
                        double[] bGrad = storageOf(b).grad64();
                        double[] aData = storageOf(a).data64();
                        double[] outGrad = outStore.grad64();
                        for (int kk = 0; kk < k; kk++) {
                            for (int j = 0; j < n; j++) {
                                double sum = 0.0;
                                for (int i = 0; i < m; i++) {
                                    sum += aData[i * k + kk] * outGrad[i * n + j];
                                }
                                bGrad[kk * n + j] += sum;
                            }
                        }
                    } else {
                        float[] bGrad = storageOf(b).grad32();
                        float[] aData = storageOf(a).data32();
                        float[] outGrad = outStore.grad32();
                        for (int kk = 0; kk < k; kk++) {
                            for (int j = 0; j < n; j++) {
                                float sum = 0.0f;
                                for (int i = 0; i < m; i++) {
                                    sum += aData[i * k + kk] * outGrad[i * n + j];
                                }
                                bGrad[kk * n + j] += sum;
                            }
                        }
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor embedding(Tensor weight, Tensor indices) {
        // PyTorch equivalent: torch.nn.functional.embedding. See TensorMath.md#embedding
        // Forward: gather rows from weight using integer indices
        // Backward: scatter gradients back into the same rows (summing for repeats)
        if (indices.rows() < 1 || indices.cols() < 1) {
            throw new IllegalArgumentException("Indices tensor must be non-empty for embedding lookup");
        }
        if (weight.cols() < 1) {
            throw new IllegalArgumentException("Embedding matrix must have >= 1 column");
        }
        if (weight.device() != indices.device()) {
            throw new IllegalArgumentException("Embedding weight and indices must live on same device");
        }

        int blockCount = indices.rows() * indices.cols();
        int embedDim = weight.cols();
        boolean requiresGrad = weight.requiresGrad();
        Tensor out = new Tensor(blockCount, embedDim, weight.precision(), weight.device(), requiresGrad);
        CpuTensorStorage outStore = storageOf(out);
        int[] gatheredIdx = new int[blockCount];

        if (weight.precision() == Precision.FP64) {
            double[] weightData = storageOf(weight).data64();
            double[] outData = outStore.data64();
            for (int i = 0; i < blockCount; i++) {
                int idx = readIndex(indices, i);
                if (idx < 0 || idx >= weight.rows()) {
                    throw new IllegalArgumentException(
                        String.format("Embedding index %d out of range [0, %d)", idx, weight.rows()));
                }
                gatheredIdx[i] = idx;
                int weightOffset = idx * embedDim;
                int outOffset = i * embedDim;
                System.arraycopy(weightData, weightOffset, outData, outOffset, embedDim);
            }
        } else {
            float[] weightData = storageOf(weight).data32();
            float[] outData = outStore.data32();
            for (int i = 0; i < blockCount; i++) {
                int idx = readIndex(indices, i);
                if (idx < 0 || idx >= weight.rows()) {
                    throw new IllegalArgumentException(
                        String.format("Embedding index %d out of range [0, %d)", idx, weight.rows()));
                }
                gatheredIdx[i] = idx;
                int weightOffset = idx * embedDim;
                int outOffset = i * embedDim;
                System.arraycopy(weightData, weightOffset, outData, outOffset, embedDim);
            }
        }

        if (requiresGrad) {
            out.setParents(weight);
            out.setOp("embedding");
            out.setBackwardFn(() -> {
                if (!weight.requiresGrad()) {
                    return;
                }
                int weightRows = weight.rows();
                if (weight.precision() == Precision.FP64) {
                    double[] weightGrad = storageOf(weight).grad64();
                    double[] outGrad = outStore.grad64();
                    for (int i = 0; i < blockCount; i++) {
                        int idx = gatheredIdx[i];
                        if (idx < 0 || idx >= weightRows) {
                            continue;
                        }
                        int weightOffset = idx * embedDim;
                        int outOffset = i * embedDim;
                        for (int d = 0; d < embedDim; d++) {
                            weightGrad[weightOffset + d] += outGrad[outOffset + d];
                        }
                    }
                } else {
                    float[] weightGrad = storageOf(weight).grad32();
                    float[] outGrad = outStore.grad32();
                    for (int i = 0; i < blockCount; i++) {
                        int idx = gatheredIdx[i];
                        if (idx < 0 || idx >= weightRows) {
                            continue;
                        }
                        int weightOffset = idx * embedDim;
                        int outOffset = i * embedDim;
                        for (int d = 0; d < embedDim; d++) {
                            weightGrad[weightOffset + d] += outGrad[outOffset + d];
                        }
                    }
                }
            });
        }

        return out;
    }

    @Override
    public Tensor crossEntropy(Tensor logits, Tensor targets) {
        // PyTorch equivalent: torch.nn.functional.cross_entropy. See TensorMath.md#cross-entropy
        // Forward: stable log-softmax + negative log likelihood, averaged over batch
        // Backward: softmax(logits) - one_hot(target)
        if (logits.rows() < 1 || logits.cols() < 1) {
            throw new IllegalArgumentException("Logits tensor must be non-empty for cross-entropy");
        }
        if (targets.rows() != logits.rows() || targets.cols() != 1) {
            throw new IllegalArgumentException(
                "Targets must have shape (batch, 1) and match logits batch size");
        }
        if (logits.device() != targets.device()) {
            throw new IllegalArgumentException("Logits and targets must be on same device");
        }

        boolean requiresGrad = logits.requiresGrad();
        Tensor out = new Tensor(1, 1, logits.precision(), logits.device(), requiresGrad);
        CpuTensorStorage outStore = storageOf(out);
        int batch = logits.rows();
        int classes = logits.cols();

        if (logits.precision() == Precision.FP64) {
            double[] logitsData = storageOf(logits).data64();
            double[] outData = outStore.data64();
            double[] probs = new double[logitsData.length];
            int[] targetIdx = readTargetIndices(targets);

            double lossSum = 0.0;
            for (int i = 0; i < batch; i++) {
                int rowOffset = i * classes;
                double maxLogit = logitsData[rowOffset];
                for (int j = 1; j < classes; j++) {
                    maxLogit = Math.max(maxLogit, logitsData[rowOffset + j]);
                }
                double sumExp = 0.0;
                for (int j = 0; j < classes; j++) {
                    double exp = Math.exp(logitsData[rowOffset + j] - maxLogit);
                    probs[rowOffset + j] = exp;
                    sumExp += exp;
                }
                for (int j = 0; j < classes; j++) {
                    probs[rowOffset + j] /= sumExp;
                }
                int t = targetIdx[i];
                if (t < 0 || t >= classes) {
                    throw new IllegalArgumentException(
                        String.format("Target index %d out of range [0, %d)", t, classes));
                }
                lossSum += -Math.log(Math.max(probs[rowOffset + t], 1e-12));
            }

            double meanLoss = lossSum / batch;
            outData[0] = meanLoss;

            if (requiresGrad) {
                out.setParents(logits);
                out.setOp("cross_entropy");
                out.setBackwardFn(() -> {
                    double[] logitsGrad = storageOf(logits).grad64();
                    double[] outGrad = outStore.grad64();
                    double upstream = outGrad[0] / batch;
                    for (int i = 0; i < batch; i++) {
                        int rowOffset = i * classes;
                        for (int j = 0; j < classes; j++) {
                            double grad = probs[rowOffset + j];
                            if (j == targetIdx[i]) {
                                grad -= 1.0;
                            }
                            logitsGrad[rowOffset + j] += grad * upstream;
                        }
                    }
                });
            }
        } else {
            float[] logitsData = storageOf(logits).data32();
            float[] outData = outStore.data32();
            float[] probs = new float[logitsData.length];
            int[] targetIdx = readTargetIndices(targets);

            double lossSum = 0.0;
            for (int i = 0; i < batch; i++) {
                int rowOffset = i * classes;
                float maxLogit = logitsData[rowOffset];
                for (int j = 1; j < classes; j++) {
                    maxLogit = Math.max(maxLogit, logitsData[rowOffset + j]);
                }
                float sumExp = 0.0f;
                for (int j = 0; j < classes; j++) {
                    float exp = (float) Math.exp(logitsData[rowOffset + j] - maxLogit);
                    probs[rowOffset + j] = exp;
                    sumExp += exp;
                }
                for (int j = 0; j < classes; j++) {
                    probs[rowOffset + j] /= sumExp;
                }
                int t = targetIdx[i];
                if (t < 0 || t >= classes) {
                    throw new IllegalArgumentException(
                        String.format("Target index %d out of range [0, %d)", t, classes));
                }
                lossSum += -Math.log(Math.max(probs[rowOffset + t], 1e-12f));
            }

            float meanLoss = (float) (lossSum / batch);
            outData[0] = meanLoss;

            if (requiresGrad) {
                out.setParents(logits);
                out.setOp("cross_entropy");
                out.setBackwardFn(() -> {
                    float[] logitsGrad = storageOf(logits).grad32();
                    float[] outGrad = outStore.grad32();
                    float upstream = outGrad[0] / batch;
                    for (int i = 0; i < batch; i++) {
                        int rowOffset = i * classes;
                        for (int j = 0; j < classes; j++) {
                            float grad = probs[rowOffset + j];
                            if (j == targetIdx[i]) {
                                grad -= 1.0f;
                            }
                            logitsGrad[rowOffset + j] += grad * upstream;
                        }
                    }
                });
            }
        }

        return out;
    }

    @Override
    public Tensor reshape(Tensor input, int newRows, int newCols) {
        // PyTorch equivalent: tensor.view(newRows, newCols). See TensorMath.md#reshape
        // Forward: reinterprets the flat buffer with new shape
        // Backward: gradient is reshaped back to original form (identity)
        if (input.rows() * input.cols() != newRows * newCols) {
            throw new IllegalArgumentException(
                String.format("Cannot reshape (%d, %d) into (%d, %d)",
                    input.rows(), input.cols(), newRows, newCols));
        }

        Tensor out = new Tensor(newRows, newCols, input.precision(), input.device(), input.requiresGrad());
        CpuTensorStorage inStore = storageOf(input);
        CpuTensorStorage outStore = storageOf(out);
        if (input.precision() == Precision.FP64) {
            double[] src = inStore.data64();
            double[] dst = outStore.data64();
            System.arraycopy(src, 0, dst, 0, src.length);
        } else {
            float[] src = inStore.data32();
            float[] dst = outStore.data32();
            System.arraycopy(src, 0, dst, 0, src.length);
        }

        if (input.requiresGrad()) {
            out.setParents(input);
            out.setOp("reshape");
            out.setBackwardFn(() -> {
                if (input.precision() == Precision.FP64) {
                    double[] inGrad = inStore.grad64();
                    double[] outGrad = outStore.grad64();
                    accumulate(inGrad, outGrad);
                } else {
                    float[] inGrad = inStore.grad32();
                    float[] outGrad = outStore.grad32();
                    accumulate(inGrad, outGrad);
                }
            });
        }

        return out;
    }

    @Override
    public void sgdStep(Tensor param, double lr) {
        if (!param.requiresGrad()) {
            throw new IllegalStateException("Cannot run SGD on tensor without gradients");
        }
        CpuTensorStorage storage = storageOf(param);
        if (param.precision() == Precision.FP64) {
            double[] data = storage.data64();
            double[] grad = storage.grad64();
            for (int i = 0; i < data.length; i++) {
                data[i] -= lr * grad[i];
            }
        } else {
            float[] data = storage.data32();
            float[] grad = storage.grad32();
            for (int i = 0; i < data.length; i++) {
                data[i] -= (float) (lr * grad[i]);
            }
        }
    }

    private void broadcastReduceGrad(Tensor x, Tensor out, int axis, double scale) {
        if (x.precision() == Precision.FP64) {
            double[] xGrad = storageOf(x).grad64();
            double[] outGrad = storageOf(out).grad64();
            if (axis == 0) {
                for (int r = 0; r < x.rows(); r++) {
                    for (int c = 0; c < x.cols(); c++) {
                        xGrad[r * x.cols() + c] += outGrad[c] * scale;
                    }
                }
            } else {
                for (int r = 0; r < x.rows(); r++) {
                    for (int c = 0; c < x.cols(); c++) {
                        xGrad[r * x.cols() + c] += outGrad[r] * scale;
                    }
                }
            }
        } else {
            float[] xGrad = storageOf(x).grad32();
            float[] outGrad = storageOf(out).grad32();
            float scaleF = (float) scale;
            if (axis == 0) {
                for (int r = 0; r < x.rows(); r++) {
                    for (int c = 0; c < x.cols(); c++) {
                        xGrad[r * x.cols() + c] += outGrad[c] * scaleF;
                    }
                }
            } else {
                for (int r = 0; r < x.rows(); r++) {
                    for (int c = 0; c < x.cols(); c++) {
                        xGrad[r * x.cols() + c] += outGrad[r] * scaleF;
                    }
                }
            }
        }
    }

    private CpuTensorStorage storageOf(Tensor tensor) {
        if (tensor.device() != DeviceType.CPU) {
            throw new IllegalArgumentException(
                "Tensor resides on " + tensor.device() + " but CpuBackend only handles CPU tensors");
        }
        return (CpuTensorStorage) tensor.storage();
    }

    private void ensureBinaryCompat(String op, Tensor a, Tensor b) {
        ensureSameDeviceAndPrecision(op, a, b);
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw new IllegalArgumentException(
                String.format("%s Shape mismatch: (%d, %d) vs (%d, %d)",
                    op, a.rows(), a.cols(), b.rows(), b.cols()));
        }
    }

    private void ensureSameDeviceAndPrecision(String op, Tensor a, Tensor b) {
        if (a.device() != b.device()) {
            throw new IllegalArgumentException(
                String.format("%s requires tensors on same device, got %s vs %s",
                    op, a.device(), b.device()));
        }
        if (a.precision() != b.precision()) {
            throw new IllegalArgumentException(
                String.format("%s requires tensors with same precision, got %s vs %s",
                    op, a.precision(), b.precision()));
        }
    }

    private static void accumulate(double[] target, double[] source) {
        for (int i = 0; i < target.length; i++) {
            target[i] += source[i];
        }
    }

    private static void accumulate(float[] target, float[] source) {
        for (int i = 0; i < target.length; i++) {
            target[i] += source[i];
        }
    }

    private int readIndex(Tensor indices, int flatIndex) {
        if (indices.precision() == Precision.FP64) {
            double[] data = storageOf(indices).data64();
            return (int) Math.round(data[flatIndex]);
        } else {
            float[] data = storageOf(indices).data32();
            return Math.round(data[flatIndex]);
        }
    }

    private int[] readTargetIndices(Tensor targets) {
        int batch = targets.rows();
        int[] out = new int[batch];
        if (targets.precision() == Precision.FP64) {
            double[] data = storageOf(targets).data64();
            for (int i = 0; i < batch; i++) {
                out[i] = (int) Math.round(data[i * targets.cols()]);
            }
        } else {
            float[] data = storageOf(targets).data32();
            for (int i = 0; i < batch; i++) {
                out[i] = Math.round(data[i * targets.cols()]);
            }
        }
        return out;
    }
}


