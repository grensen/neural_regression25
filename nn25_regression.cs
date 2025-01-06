// https://github.com/grensen/neural_regression25
// .NET 9
// neural network regression demo, supported optimizers: sgd, adam
// supported data sets: people, optimization, synthetic
// nn24 -> nn 25: optimized signal flow (input to output), fixed momentum,
// new basic implementation: (pre acitivation, nodes to gradients, no input layer gradients)
// new evaluation score, standartized for ml models
// tipp: cntrl + m + o

public class Program
{
    static void Main()
    {
        Console.WriteLine($"\n" + ("NeuralNet") +
            $" regression {DateTime.Now.Year} C# (.NET {Environment.Version})");

        var dataset = Data.DatasetType.People;
        Data dt = new Data(dataset);

        Console.WriteLine($"\nLoading {dataset} train ({dt.YTrain.Length}) and test ({dt.YTest.Length}) data");
    
        Console.WriteLine("\nFirst three train X: ");
        for (int i = 0; i < 3; ++i) Data.VecShow(dt.XTrain[i], 4, 8);
        Console.WriteLine("\nFirst three train y: ");
        for (int i = 0; i < 3; ++i)
            Console.Write($"{dt.YTrain[i],8:F4}");
        Console.WriteLine("");

        int inputs = dt.XTrain[0].Length;
        int[] net = [inputs, 100, 100, 1];
        int EPOCHS = 1000; // training data * epochs
        int BATCHSIZE = 50; // samples till update
        float LEARNINGRATE = 0.005f; // learning imapact
        float MOMENTUM = 0.98f; // learning memory (sgd only)
        float DECAY = 0.005f; // weight decay each update
        float FACTOR = 0.999f; // decrease of (lr and decay) each epoch
        float BP = 0.001f; // error threshold regularization

        bool ADAM = false; // optimizer adam or sgd
        float BETA1 = 0.9f;
        float BETA2 = 0.99999f;
        float EPSILON = 0.0001f;

        var sw = System.Diagnostics.Stopwatch.StartNew();

        var nn = EvaluateModel(new NeuralNet25(net, 1337),
            m => m.Train(dt.XTrain, dt.YTrain,
            LEARNINGRATE, MOMENTUM, BETA1, BETA2, EPSILON, DECAY,
            FACTOR, EPOCHS, BATCHSIZE, BP, ADAM)
            , dt.XTrain, dt.YTrain, dt.XTest, dt.YTest);

        Console.WriteLine($"\nFirst three predicted y = ");
        for (int i = 0; i < 3; i++)
            Console.Write($"{nn.Predict(dt.XTrain[i]),8:F4}");
        Console.WriteLine("");

        sw.Stop();
        Console.WriteLine($"\nEnd demo after {sw.Elapsed.TotalMilliseconds / 1000.0:F3}s");
        
    } // Main

    public static T EvaluateModel<T>(T model, Action<T> trainAction,
    float[][] XTrain, float[] YTrain, float[][] XTest, float[] YTest, bool raw = true)
    {
        trainAction(model);
        dynamic dynamicModel = model; // Use dynamic to invoke methods on the model.

        try
        {
            // Compute metrics
            var trainMSE = RootMSE(XTrain, YTrain, dynamicModel);
            var trainMAE = MAE(XTrain, YTrain, dynamicModel);
            var trainAcc1 = Accuracy(XTrain, YTrain, 0.01f, dynamicModel, raw);
            var trainAcc5 = Accuracy(XTrain, YTrain, 0.05f, dynamicModel, raw);
            var trainAcc10 = Accuracy(XTrain, YTrain, 0.1f, dynamicModel, raw);

            var testMSE = RootMSE(XTest, YTest, dynamicModel);
            var testMAE = MAE(XTest, YTest, dynamicModel);
            var testAcc1 = Accuracy(XTest, YTest, 0.01f, dynamicModel, raw);
            var testAcc5 = Accuracy(XTest, YTest, 0.05f, dynamicModel, raw);
            var testAcc10 = Accuracy(XTest, YTest, 0.1f, dynamicModel, raw);

            // Output results
            float scoreTest = (1 - ((testAcc1 * 3 + testAcc5 * 2 + testAcc10) / 6)) * ((testMSE + testMAE) / 2) * 100;
            float scoreTrain = (1 - ((trainAcc1 * 3 + trainAcc5 * 2 + trainAcc10) / 6)) * ((trainMSE + trainMAE) / 2) * 100;

            if (scoreTrain >= 10) scoreTrain = 9.999f;
            if (scoreTest >= 10) scoreTest = 9.999f;

            Console.Write($"\nTrained {dynamicModel.GetType().Name} evaluation\n");

            Console.WriteLine($"\nError(RMSE, MAE), AccuracyWithinRaw(10%, 5%, 1%),");
            Console.WriteLine($"Score((RMSE + MAE) / 2) * (1 - (10% + 2*5% + 3*1%) / 6))");

            Console.WriteLine(
                $"\nTraining: " +
                $"E({trainMSE:F3}, {trainMAE:F3}), " +
                $"A({trainAcc10:F2}, {trainAcc5:F2}, {trainAcc1:F2}), " +
                $"S({scoreTrain:F3})" + $"" +
                $"\nTesting:  " +
                $"E({testMSE:F3}, {testMAE:F3}), " +
                $"A({testAcc10:F2}, {testAcc5:F2}, {testAcc1:F2}), " +
                $"S({scoreTest:F3}) ");
        }
        catch (Microsoft.CSharp.RuntimeBinder.RuntimeBinderException ex)
        {
            Console.WriteLine($"Error evaluating model '{dynamicModel.GetType().Name}: {ex.Message}'");
        }

        // Metrics functions
        float RootMSE(float[][] dataX, float[] dataY, dynamic model)
        {
            float sum = 0.0f;
            for (int i = 0; i < dataX.Length; ++i)
            {
                float yActual = dataY[i];
                float yPred = model.Predict(dataX[i]);
                sum += (yActual - yPred) * (yActual - yPred);
            }
            return MathF.Sqrt(sum / dataX.Length);
        }

        float MAE(float[][] dataX, float[] dataY, dynamic model)
        {
            float sum = 0.0f;
            for (int i = 0; i < dataX.Length; ++i)
            {
                float yActual = dataY[i];
                float yPred = model.Predict(dataX[i]);
                sum += MathF.Abs(yActual - yPred);
            }
            return sum / dataX.Length;
        }

        float Accuracy(float[][] data, float[] labels, float pctClose, dynamic model, bool raw = true)
        {
            int correct = 0;
            for (int i = 0; i < labels.Length; i++)
            {
                float prediction = model.Predict(data[i]);
                float target = labels[i];
                float dist = raw ? pctClose : MathF.Abs(pctClose * target);
                if (Math.Abs(target - prediction) < dist)
                    ++correct;
            }
            return correct / (float)labels.Length;
        }
        return model;
    }

} // Program

public class NeuralNet25
{
    public int SEED;
    public int[] net;
    public float[][] wt;
    public float[] bias;

    public NeuralNet25(int[] _net, int seed) => InitNetwork(_net, seed);

    public NeuralNet25(int inputs, int LAYERS, int NEURONS, int outputs, int seed)
    {
        int[] hiddenLayer = new int[LAYERS];
        for (int i = 0; i < hiddenLayer.Length; i++)
            hiddenLayer[i] = NEURONS;
        InitNetwork([inputs, .. hiddenLayer, outputs], seed);
    }

    void InitNetwork(int[] _net, int seed)
    {
        net = _net;

        bias = new float[net.Sum() - net[0]];
        Random rnd = new Random(seed);

        if (false)
            for (int i = 0; i < bias.Length; i++)
                bias[i] += (float)(rnd.NextDouble() - 0.5f) * 0.1f;

        wt = CreateJaggedArray<float>(net);
        SEED = seed;
        WeightInit(net, wt, SEED);
    }

    public void Train(float[][] data, float[] labels, float lr, float mom,
        float beta1, float beta2, float epsilon, float decay, float factor, 
        int epochs, int batch, float bp, bool adam, bool multiCore = false, bool info = true)
    {

        Info();

        int len = (labels.Length / batch) * batch;
        int networkSize = net.Sum();

        var rng = new Random(SEED);
        int[] indices = new int[len];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;

        float[] biasDelta = new float[net.Sum() - net[0]];
        float[] mBias = new float[bias.Length], vBias = new float[bias.Length];
        float[][] deltas = CreateJaggedArray<float>(net);
        float[][] mW = CreateJaggedArray<float>(net);
        float[][] vW = CreateJaggedArray<float>(net);

        for (int epoch = 0, B = len / batch; epoch < epochs; epoch++,
            lr *= factor, decay *= factor)
        {
            Shuffle(indices, rng);
            for (int b = 0; b < B; b++)
            {
                if (multiCore)
                    Parallel.For(b * batch, (b + 1) * batch, x =>
                    {
                        TrainSample(indices[x]);
                    });
                else
                    for (int x = b * batch, X = (b + 1) * batch; x < X; x++)
                        TrainSample(indices[x]);

                if (adam)
                    Adam(bias, biasDelta, wt, deltas, lr, beta1, beta2, epsilon, decay, mBias, vBias, mW, vW, epoch + 1);
                else
                    SGD(bias, biasDelta, wt, deltas, lr, mom, decay);
            }
        }

        void TrainSample(int id)
        {
            float target = labels[id];
            Span<float> neurons = stackalloc float[networkSize];
            // Span<float> inputs = data[id];
            data[id].AsSpan().CopyTo(neurons);
            float prediction = FeedForward(net, neurons, bias, wt);

            // check if error is large enough for bp
            if (Math.Abs(target - prediction) > bp)
            {
                // output gradient
                for (int i = 0; i < net[^1]; i++)
                    neurons[^(1 + i)] = target - neurons[^(1 + i)];

                Backprop(net, neurons, biasDelta, wt, deltas);
            }
        }

        void Info()
        {
            if (info)
            {
                Console.WriteLine($"\nNeuralNet with {(adam ? "ADAM" : "SGD")}: ");
                Console.WriteLine("NETWORK      = " + string.Join("-", net));
                Console.WriteLine("WEIGHTS      = " + this.WeightSize());
                Console.WriteLine("LEARNINGRATE = " + lr);
                if (adam)
                {
                    Console.WriteLine("BETA1        = " + beta1);
                    Console.WriteLine("BETA2        = " + beta2);
                    Console.WriteLine("EPSILON      = " + epsilon);
                }
                else
                    Console.WriteLine("MOMENTUM     = " + mom);
                Console.WriteLine("WEIGHTDECAY  = " + decay);
                Console.WriteLine("FACTOR       = " + factor);
                Console.WriteLine("BATCHSIZE    = " + batch);
                Console.WriteLine("EPOCHS       = " + epochs);
                Console.WriteLine("BP           = " + bp);
            }
        }
    }

    public static float FeedForward(int[] net, Span<float> neurons, float[] bias, float[][] weights)
    {
        for (int i = 0, k = 0, input = net[0]; i < net.Length - 1; i++)
        {
            int j = k; k += net[i];
            if (i == 0) // works with negative inputs 
                for (int jl = j; jl < k; jl++)
                {
                    float n = neurons[jl];
                    if (n != 0)
                        for (int R = weights[jl].Length, r = 0; r < R; r++)
                            neurons[k + r] += weights[jl][r] * n;
                }
            else
                for (int jl = j; jl < k; jl++)
                {
                    float n = neurons[jl];
                    if (n > 0) // Pre-ReLU
                        for (int R = weights[jl].Length, r = 0; r < R; r++)
                            neurons[k + r] += weights[jl][r] * n;
                }
            // add bias last
            for (int r = 0; r < net[i + 1]; r++)
                neurons[k + r] += bias[k + r - input];
        }
        // simple mean if more than one output (indentity)
        float sumPredictions = 0;
        for (int i = 0; i < net[^1]; i++)
            sumPredictions += neurons[^(1 + i)];

        return sumPredictions / net[^1];
    }

    public static void Backprop(int[] net, Span<float> neurons, float[] biasDelta, float[][] weights, float[][] deltas)
    {
        int input = net[0], outLayerStart = neurons.Length - net[^1];
        for (int j = outLayerStart, i = net.Length - 2; i >= 0; i--)
        {
            int k = j; j -= net[i];
            for (int r = 0; r < net[i + 1]; r++)
                biasDelta[k + r - input] += neurons[k + r];

            if (i != 0) // hidden layer
                for (int jl = j; jl < k; jl++)
                {
                    float n = neurons[jl], sumGradient = 0;
                    if (n > 0)  // Pre-ReLU
                        for (int R = weights[jl].Length, r = 0; r < R; r++)
                        {
                            float gradient = neurons[k + r];
                            sumGradient += weights[jl][r] * gradient;
                            deltas[jl][r] += n * gradient;
                        }
                    neurons[jl] = sumGradient; // reusing neurons for gradient computation
                }
            else // input layer
                for (int jl = j; jl < k; jl++)
                {
                    float n = neurons[jl];
                    if (n != 0)  // Pre-ReLU
                        for (int R = weights[jl].Length, r = 0; r < R; r++)
                            deltas[jl][r] += n * neurons[k + r];
                }
        }
    }

    public static void SGD(float[] bias, float[] biasDelta, float[][] weights, float[][] deltas, float lr, float mom, float decay)
    {
        for (int i = 0; i < bias.Length; i++) // no bias decay !!!
        {
            bias[i] += biasDelta[i] * lr;
            biasDelta[i] *= mom;
        }

        for (int i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].Length; j++)
            {
                var wt = weights[i][j] - weights[i][j] * decay;
                weights[i][j] = wt + deltas[i][j] * lr;
                deltas[i][j] *= mom;
            }
    }

    public static void Adam(float[] bias, float[] biasDelta, float[][] weights, float[][] deltas,
        float lr, float beta1, float beta2, float epsilon, float weightDecay,
        float[] mBias, float[] vBias, float[][] mW, float[][] vW, int t)
    {
        // Adam for bias (NO weight decay)
        for (int i = 0; i < bias.Length; i++)
        {
            mBias[i] = beta1 * mBias[i] + (1 - beta1) * biasDelta[i];
            vBias[i] = beta2 * vBias[i] + (1 - beta2) * biasDelta[i] * biasDelta[i];
            float mHat = mBias[i] / (1 - (float)Math.Pow(beta1, t));
            float vHat = vBias[i] / (1 - (float)Math.Pow(beta2, t));
            bias[i] += lr * mHat / (float)Math.Sqrt(vHat + epsilon); // No weight decay
            biasDelta[i] = 0;
        }

        // Adam for weights (WITH weight decay)
        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < weights[i].Length; j++)
            {
                mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * deltas[i][j];
                vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * deltas[i][j] * deltas[i][j];
                float mHat = mW[i][j] / (1 - (float)Math.Pow(beta1, t));
                float vHat = vW[i][j] / (1 - (float)Math.Pow(beta2, t));
                weights[i][j] += lr * mHat / (float)Math.Sqrt(vHat + epsilon) - lr * weightDecay * weights[i][j]; // Weight decay applied
                deltas[i][j] = 0;
            }
        }
    }

    public float Predict(float[] sample)
    {
        var neurons = new float[net.Sum()];
        for (int j = 0; j < net[0]; j++) neurons[j] = sample[j];
        return FeedForward(net, neurons, bias, wt);
    }

    public static void Shuffle(int[] arr, Random rnd)
    {
        // Fisher-Yates algorithm
        int n = arr.Length;
        for (int i = 0; i < n; ++i)
        {
            int ri = rnd.Next(i, n);  // random index
            (arr[i], arr[ri]) = (arr[ri], arr[i]);
        }
    }

    public static void WeightInit(int[] net, float[][] weights, int seed)
    {
        Random rnd = new Random(seed);
        for (int j = 0, k = net[0], i = 0; i < net.Length - 1; i++)
        {
            float sd = MathF.Sqrt(6.0f / (net[i] + weights[j + 0].Length));
            for (int l = 0; l < net[i]; l++)
                for (int r = 0; r < weights[j + l].Length; r++)
                    weights[j + l][r] = (float)(rnd.NextDouble() * sd * 2 - sd) * 0.5f;
            j += net[i]; k += net[i + 1];
        }
    }

    public int WeightSize() => GetWeightsSize(wt) + bias.Length;

    public static int GetWeightsSize<T>(T[][] array)
    {
        int size = 0;
        for (int i = 0; i < array.Length; i++)
            size += array[i].Length;
        return size;
    }

    public static T[][] CreateJaggedArray<T>(int[] net)
    {
        T[][] array = new T[net.Sum() - net[^1]][];
        for (int i = 0, j = 0; i < net.Length - 1; i++)
        {
            for (int l = 0; l < net[i]; l++)
                array[j + l] = new T[net[i + 1]];
            j += net[i];
        }
        return array;
    }
}

public class Data
{
    public enum DatasetType { People, Synthetic, Optimization }
    public int Inputs { get; }
    public float[][] XTrain { get; }
    public float[] YTrain { get; }
    public float[][] XTest { get; }
    public float[] YTest { get; }

    public Data(DatasetType set, int trainLen = 200, int trainSeed = 1337, int testLen = 100, int testSeed = 12345)
    {
        switch (set)
        {
            case DatasetType.Synthetic:
                Inputs = 5;
                XTrain = Extract(TrainDataSynth24(), [0, 1, 2, 3, 4]);
                YTrain = ExtractVec(TrainDataSynth24(), 5);
                XTest = Extract(TestDataSynth24(), [0, 1, 2, 3, 4]);
                YTest = ExtractVec(TestDataSynth24(), 5);
                break;

            case DatasetType.People:
                Inputs = 8;
                XTrain = Extract(TrainDataPeople24(), [0, 1, 2, 3, 4, 6, 7, 8]);
                YTrain = ExtractVec(TrainDataPeople24(), 5);
                XTest = Extract(TestDataPeople24(), [0, 1, 2, 3, 4, 6, 7, 8]);
                YTest = ExtractVec(TestDataPeople24(), 5);
                break;

            case DatasetType.Optimization:
                int features = 7;
                bool isLabelNorm = true;
                float targetMax = 100.0f;

                // range init
                float[] minVal = new float[features];
                float[] maxVal = new float[features];
                //
                minVal[0] = -0.000f;
                maxVal[0] = 5;
                // exps
                minVal[1] = 50;
                maxVal[1] = 200;
                // lr
                minVal[2] = 0.01f;
                maxVal[2] = 0.99f;
                // decay
                minVal[3] = 0.9f;
                maxVal[3] = 1;
                // tol
                minVal[4] = 0.00000001f;
                maxVal[4] = 0.01f;
                // gra
                minVal[5] = 0.00000001f;
                maxVal[5] = 0.001f;
                // hps
                minVal[6] = 10;
                maxVal[6] = 12;

                float[] data = CreateData(features, trainLen, minVal, maxVal, trainSeed, targetMax);
                float[] data2 = CreateData(features, testLen, minVal, maxVal, testSeed, targetMax);

                Inputs = 6;
                XTrain = CreateNormData(data.Length / features, features, data, isLabelNorm);
                YTrain = CreateLabels(data.Length / features, features, data, isLabelNorm);
                XTest = CreateNormData(data2.Length / features, features, data2, isLabelNorm);
                YTest = CreateLabels(data2.Length / features, features, data2, isLabelNorm);
                break;
            default:
                throw new ArgumentException("Invalid dataset type.");
        }
    }

    public static void VecShow(float[] vec, int dec, int wid)
    {
        for (int i = 0; i < vec.Length; ++i)
            Console.Write(vec[i].ToString("F" + dec).
              PadLeft(wid));
        Console.WriteLine("");
    }
    public static float[] CreateData(int features, int dataLen, float[] minVal, float[] maxVal, int seed, float targetMax)
    {
        float[] arr = new float[dataLen * features];
        float[] trainData = [];
        Random random = new Random(seed);
        for (int experiments = 0; experiments < dataLen; experiments++)
        {
            int maxExperiments = GetRandomInt((int)minVal[1], (int)maxVal[1]);

            float learningRate = GetRandomFloat(minVal[2], maxVal[2]);

            float learningRateDecay = GetRandomFloat(minVal[3], maxVal[3]);

            float tolerance = GetRandomFloat(minVal[4], maxVal[4]);

            float gradient = GetRandomFloat(minVal[5], maxVal[5]);

            int numHyperparameters = GetRandomInt((int)minVal[6], (int)maxVal[6]);

            float target = Optimizer(maxExperiments, learningRate, learningRateDecay, tolerance, gradient, numHyperparameters);

            if (targetMax < target)
            {
                experiments--; continue;
            }
            int id = experiments * features;
            arr[id + 0] = target;
            arr[id + 1] = maxExperiments;
            arr[id + 2] = learningRate;
            arr[id + 3] = learningRateDecay;
            arr[id + 4] = tolerance;
            arr[id + 5] = gradient;
            arr[id + 6] = numHyperparameters;
        }

        trainData = [.. trainData, .. arr];
        return trainData;
        float GetRandomFloat(float minValue, float maxValue) => random.NextSingle() * (maxValue - minValue) + minValue;
        int GetRandomInt(int minValue, int maxValue) => random.Next(minValue, maxValue + 1);
    }
    public static float[] CreateLabels(int trainingLen, int features, float[] data, bool isLabelNorm)
    {
        float[] labels = new float[trainingLen];
        for (int i = 0; i < trainingLen; i++)
            labels[i] = data[i * features];

        float min = float.MaxValue, max = float.MinValue;
        // Find the minimum and maximum values in the array
        foreach (float value in labels)
        {
            if (value < min) min = value;
            if (value > max) max = value;
        }
        for (int i = 0; i < trainingLen; i++)
            labels[i] = LabelNorm((labels[i] - min) / (max - min), isLabelNorm);

        return labels;
    }

    public static float[][] CreateNormData(int trainingLen, int features, float[] data, bool isLabelNorm)
    {
        float[][] dataNorm = new float[trainingLen][];
        for (int i = 0; i < trainingLen; i++)
        {
            dataNorm[i] = new float[features - 1];
            for (int j = 1; j < features; j++)
                dataNorm[i][j - 1] = data[i * features + j];
        }
        // Find min and max for each column
        float[] minValues = new float[dataNorm[0].Length];
        float[] maxValues = new float[dataNorm[0].Length];
        for (int j = 0; j < dataNorm[0].Length; j++)
        {
            minValues[j] = float.MaxValue;
            maxValues[j] = float.MinValue;
            for (int i = 0; i < trainingLen; i++)
            {
                if (dataNorm[i][j] < minValues[j]) minValues[j] = dataNorm[i][j];
                if (dataNorm[i][j] > maxValues[j]) maxValues[j] = dataNorm[i][j];
            }
        }
        // Normalize the data
        for (int i = 0; i < trainingLen; i++)
            for (int j = 0; j < dataNorm[0].Length; j++)
                dataNorm[i][j] = NormData(dataNorm[i][j], minValues[j], maxValues[j]); // ((trainingData[i][j] - minValues[j]) / (maxValues[j] - minValues[j]) - 0.0f) * 1;               
        return dataNorm;
    }
    public static float LabelNorm(float output, bool invert = true) => (invert ? (1 - output) : (output));
    public static float NormData(float val, float min, float max) => (((val - min) / (max - min)) * 2 - 1);

    public static float Optimizer//Old
        (int maxExperiments, float learningRate, float learningRateDecay
    , float tolerance, float gradient, int numHyperparameters)
    {
        float[] hyperparameters = new float[numHyperparameters];
        float[] previousHyperparameters = new float[numHyperparameters];
        float[] gradients = new float[numHyperparameters];
        float previousError = float.MaxValue;
        Random rng = new Random(123);
        // Initialize hyperparameters to 1 (for simplicity)
        for (int i = 0; i < numHyperparameters; i++)
            hyperparameters[i] = (float)(rng.NextDouble() * 10 - 5); // * 2 - 1; // 1.0;

        for (int experiment = 0; experiment < maxExperiments; experiment++)
        {
            float currentError = CalculateError(hyperparameters);

            if (experiment != 0)
            {
                // Calculate gradients (simple finite difference for this example)
                int i = (experiment + 1) % numHyperparameters; // rng.Next(0, numHyperparameters);

                float originalValue = hyperparameters[i];

                hyperparameters[i] += tolerance;
                float newError = CalculateError(hyperparameters);

                gradients[i] = (newError - currentError) / tolerance;
                hyperparameters[i] = originalValue;

                // Update hyperparameters using the gradients
                previousHyperparameters[i] = hyperparameters[i];
                hyperparameters[i] -= learningRate * gradients[i];

                // Reduce learning rate over time (optional)
                learningRate *= learningRateDecay;

                // Switch the distance based on experiment results
                if (currentError > previousError)
                    hyperparameters[i] = previousHyperparameters[i] + (gradient * gradients[i]);

                // Apply gradient clipping to avoid exploding gradients
                // if (Math.Abs(gradients[i]) > gradient) gradients[i] = Math.Sign(gradients[i]) * gradient;
            }
            previousError = currentError;
        }
        return (float)previousError;

        static float CalculateError(float[] hyperparameters)
        {
            // Hypothetical error calculation (sum of squares)
            float err = 0.0f;
            for (int i = 0; i < hyperparameters.Length; i++)
                err += MathF.Pow(hyperparameters[i], 2);
            return err;
        }
    }

    public static float[][] MatCreate(int rows, int cols)
    {
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; ++i)
            result[i] = new float[cols];
        return result;
    }
    public static float[][] Extract(float[][] mat, int[] cols)
    {
        int nRows = mat.Length;
        int nCols = cols.Length;
        float[][] result = MatCreate(nRows, nCols);
        for (int i = 0; i < nRows; ++i)
        {
            for (int j = 0; j < nCols; ++j)  // idx into src cols
            {
                int srcCol = cols[j];
                int destCol = j;
                result[i][destCol] = mat[i][srcCol];
            }
        }
        return result;
    }
    public static float[] ExtractVec(float[][] mat, int label)
    {
        int nRows = mat.Length;
        float[] result = new float[nRows];
        for (int i = 0; i < nRows; ++i)
            result[i] = mat[i][label];
        return result;
    }
    public static float[][] TrainDataSynth24()
    {
        float[][] trainData = new float[][]
        {
            new float[] {-0.1660f,  0.4406f, -0.9998f, -0.3953f, -0.7065f,  0.4840f },  // 1
            new float[] { 0.0776f, -0.1616f,  0.3704f, -0.5911f,  0.7562f,  0.1568f },  // 2
            new float[] {-0.9452f,  0.3409f, -0.1654f,  0.1174f, -0.7192f,  0.8054f },  // 3
            new float[] { 0.9365f, -0.3732f,  0.3846f,  0.7528f,  0.7892f,  0.1345f },  // 4
            new float[] {-0.8299f, -0.9219f, -0.6603f,  0.7563f, -0.8033f,  0.7955f },  // 5
            new float[] { 0.0663f,  0.3838f, -0.3690f,  0.3730f,  0.6693f,  0.3206f },  // 6
            new float[] {-0.9634f,  0.5003f,  0.9777f,  0.4963f, -0.4391f,  0.7377f },  // 7
            new float[] {-0.1042f,  0.8172f, -0.4128f, -0.4244f, -0.7399f,  0.4801f },  // 8
            new float[] {-0.9613f,  0.3577f, -0.5767f, -0.4689f, -0.0169f,  0.6861f },  // 9
            new float[] {-0.7065f,  0.1786f,  0.3995f, -0.7953f, -0.1719f,  0.5569f },  // 10
            new float[] { 0.3888f, -0.1716f, -0.9001f,  0.0718f,  0.3276f,  0.2500f },  // 11
            new float[] { 0.1731f,  0.8068f, -0.7251f, -0.7214f,  0.6148f,  0.3297f },  // 12
            new float[] {-0.2046f, -0.6693f,  0.8550f, -0.3045f,  0.5016f,  0.2129f },  // 13
            new float[] { 0.2473f,  0.5019f, -0.3022f, -0.4601f,  0.7918f,  0.2613f },  // 14
            new float[] {-0.1438f,  0.9297f,  0.3269f,  0.2434f, -0.7705f,  0.5171f },  // 15
            new float[] { 0.1568f, -0.1837f, -0.5259f,  0.8068f,  0.1474f,  0.3307f },  // 16
            new float[] {-0.9943f,  0.2343f, -0.3467f,  0.0541f,  0.7719f,  0.5581f },  // 17
            new float[] { 0.2467f, -0.9684f,  0.8589f,  0.3818f,  0.9946f,  0.1092f },  // 18
            new float[] {-0.6553f, -0.7257f,  0.8652f,  0.3936f, -0.8680f,  0.7018f },  // 19
            new float[] { 0.8460f,  0.4230f, -0.7515f, -0.9602f, -0.9476f,  0.1996f },  // 20
            new float[] {-0.9434f, -0.5076f,  0.7201f,  0.0777f,  0.1056f,  0.5664f },  // 21
            new float[] { 0.9392f,  0.1221f, -0.9627f,  0.6013f, -0.5341f,  0.1533f },  // 22
            new float[] { 0.6142f, -0.2243f,  0.7271f,  0.4942f,  0.1125f,  0.1661f },  // 23
            new float[] { 0.4260f,  0.1194f, -0.9749f, -0.8561f,  0.9346f,  0.2230f },  // 24
            new float[] { 0.1362f, -0.5934f, -0.4953f,  0.4877f, -0.6091f,  0.3810f },  // 25
            new float[] { 0.6937f, -0.5203f, -0.0125f,  0.2399f,  0.6580f,  0.1460f },  // 26
            new float[] {-0.6864f, -0.9628f, -0.8600f, -0.0273f,  0.2127f,  0.5387f },  // 27
            new float[] { 0.9772f,  0.1595f, -0.2397f,  0.1019f,  0.4907f,  0.1611f },  // 28
            new float[] { 0.3385f, -0.4702f, -0.8673f, -0.2598f,  0.2594f,  0.2270f },  // 29
            new float[] {-0.8669f, -0.4794f,  0.6095f, -0.6131f,  0.2789f,  0.4700f },  // 30
            new float[] { 0.0493f,  0.8496f, -0.4734f, -0.8681f,  0.4701f,  0.3516f },  // 31
            new float[] { 0.8639f, -0.9721f, -0.5313f,  0.2336f,  0.8980f,  0.1412f },  // 32
            new float[] { 0.9004f,  0.1133f,  0.8312f,  0.2831f, -0.2200f,  0.1782f },  // 33
            new float[] { 0.0991f,  0.8524f,  0.8375f, -0.2102f,  0.9265f,  0.2150f },  // 34
            new float[] {-0.6521f, -0.7473f, -0.7298f,  0.0113f, -0.9570f,  0.7422f },  // 35
            new float[] { 0.6190f, -0.3105f,  0.8802f,  0.1640f,  0.7577f,  0.1056f },  // 36
            new float[] { 0.6895f,  0.8108f, -0.0802f,  0.0927f,  0.5972f,  0.2214f },  // 37
            new float[] { 0.1982f, -0.9689f,  0.1870f, -0.1326f,  0.6147f,  0.1310f },  // 38
            new float[] {-0.3695f,  0.7858f,  0.1557f, -0.6320f,  0.5759f,  0.3773f },  // 39
            new float[] {-0.1596f,  0.3581f,  0.8372f, -0.9992f,  0.9535f,  0.2071f },  // 40
            new float[] {-0.2468f,  0.9476f,  0.2094f,  0.6577f,  0.1494f,  0.4132f },  // 41
            new float[] { 0.1737f,  0.5000f,  0.7166f,  0.5102f,  0.3961f,  0.2611f },  // 42
            new float[] { 0.7290f, -0.3546f,  0.3416f, -0.0983f, -0.2358f,  0.1332f },  // 43
            new float[] {-0.3652f,  0.2438f, -0.1395f,  0.9476f,  0.3556f,  0.4170f },  // 44
            new float[] {-0.6029f, -0.1466f, -0.3133f,  0.5953f,  0.7600f,  0.4334f },  // 45
            new float[] {-0.4596f, -0.4953f,  0.7098f,  0.0554f,  0.6043f,  0.2775f },  // 46
            new float[] { 0.1450f,  0.4663f,  0.0380f,  0.5418f,  0.1377f,  0.2931f },  // 47
            new float[] {-0.8636f, -0.2442f, -0.8407f,  0.9656f, -0.6368f,  0.7429f },  // 48
            new float[] { 0.6237f,  0.7499f,  0.3768f,  0.1390f, -0.6781f,  0.2185f },  // 49
            new float[] {-0.5499f,  0.1850f, -0.3755f,  0.8326f,  0.8193f,  0.4399f },  // 50
            new float[] {-0.4858f, -0.7782f, -0.6141f, -0.0008f,  0.4572f,  0.4197f },  // 51
            new float[] { 0.7033f, -0.1683f,  0.2334f, -0.5327f, -0.7961f,  0.1776f },  // 52
            new float[] { 0.0317f, -0.0457f, -0.6947f,  0.2436f,  0.0880f,  0.3345f },  // 53
            new float[] { 0.5031f, -0.5559f,  0.0387f,  0.5706f, -0.9553f,  0.3107f },  // 54
            new float[] {-0.3513f,  0.7458f,  0.6894f,  0.0769f,  0.7332f,  0.3170f },  // 55
            new float[] { 0.2205f,  0.5992f, -0.9309f,  0.5405f,  0.4635f,  0.3532f },  // 56
            new float[] {-0.4806f, -0.4859f,  0.2646f, -0.3094f,  0.5932f,  0.3202f },  // 57
            new float[] { 0.9809f, -0.3995f, -0.7140f,  0.8026f,  0.0831f,  0.1600f },  // 58
            new float[] { 0.9495f,  0.2732f,  0.9878f,  0.0921f,  0.0529f,  0.1289f },  // 59
            new float[] {-0.9476f, -0.6792f,  0.4913f, -0.9392f, -0.2669f,  0.5966f },  // 60
            new float[] { 0.7247f,  0.3854f,  0.3819f, -0.6227f, -0.1162f,  0.1550f },  // 61
            new float[] {-0.5922f, -0.5045f, -0.4757f,  0.5003f, -0.0860f,  0.5863f },  // 62
            new float[] {-0.8861f,  0.0170f, -0.5761f,  0.5972f, -0.4053f,  0.7301f },  // 63
            new float[] { 0.6877f, -0.2380f,  0.4997f,  0.0223f,  0.0819f,  0.1404f },  // 64
            new float[] { 0.9189f,  0.6079f, -0.9354f,  0.4188f, -0.0700f,  0.1907f },  // 65
            new float[] {-0.1428f, -0.7820f,  0.2676f,  0.6059f,  0.3936f,  0.2790f },  // 66
            new float[] { 0.5324f, -0.3151f,  0.6917f, -0.1425f,  0.6480f,  0.1071f },  // 67
            new float[] {-0.8432f, -0.9633f, -0.8666f, -0.0828f, -0.7733f,  0.7784f },  // 68
            new float[] {-0.9444f,  0.5097f, -0.2103f,  0.4939f, -0.0952f,  0.6787f },  // 69
            new float[] {-0.0520f,  0.6063f, -0.1952f,  0.8094f, -0.9259f,  0.4836f },  // 70
            new float[] { 0.5477f, -0.7487f,  0.2370f, -0.9793f,  0.0773f,  0.1241f },  // 71
            new float[] { 0.2450f,  0.8116f,  0.9799f,  0.4222f,  0.4636f,  0.2355f },  // 72
            new float[] { 0.8186f, -0.1983f, -0.5003f, -0.6531f, -0.7611f,  0.1511f },  // 73
            new float[] {-0.4714f,  0.6382f, -0.3788f,  0.9648f, -0.4667f,  0.5950f },  // 74
            new float[] { 0.0673f, -0.3711f,  0.8215f, -0.2669f, -0.1328f,  0.2677f },  // 75
            new float[] {-0.9381f,  0.4338f,  0.7820f, -0.9454f,  0.0441f,  0.5518f },  // 76
            new float[] {-0.3480f,  0.7190f,  0.1170f,  0.3805f, -0.0943f,  0.4724f },  // 77
            new float[] {-0.9813f,  0.1535f, -0.3771f,  0.0345f,  0.8328f,  0.5438f },  // 78
            new float[] {-0.1471f, -0.5052f, -0.2574f,  0.8637f,  0.8737f,  0.3042f },  // 79
            new float[] {-0.5454f, -0.3712f, -0.6505f,  0.2142f, -0.1728f,  0.5783f },  // 80
            new float[] { 0.6327f, -0.6297f,  0.4038f, -0.5193f,  0.1484f,  0.1153f },  // 81
            new float[] {-0.5424f,  0.3282f, -0.0055f,  0.0380f, -0.6506f,  0.6613f },  // 82
            new float[] { 0.1414f,  0.9935f,  0.6337f,  0.1887f,  0.9520f,  0.2540f },  // 83
            new float[] {-0.9351f, -0.8128f, -0.8693f, -0.0965f, -0.2491f,  0.7353f },  // 84
            new float[] { 0.9507f, -0.6640f,  0.9456f,  0.5349f,  0.6485f,  0.1059f },  // 85
            new float[] {-0.0462f, -0.9737f, -0.2940f, -0.0159f,  0.4602f,  0.2606f },  // 86
            new float[] {-0.0627f, -0.0852f, -0.7247f, -0.9782f,  0.5166f,  0.2977f },  // 87
            new float[] { 0.0478f,  0.5098f, -0.0723f, -0.7504f, -0.3750f,  0.3335f },  // 88
            new float[] { 0.0090f,  0.3477f,  0.5403f, -0.7393f, -0.9542f,  0.4415f },  // 89
            new float[] {-0.9748f,  0.3449f,  0.3736f, -0.1015f,  0.8296f,  0.4358f },  // 90
            new float[] { 0.2887f, -0.9895f, -0.0311f,  0.7186f,  0.6608f,  0.2057f },  // 91
            new float[] { 0.1570f, -0.4518f,  0.1211f,  0.3435f, -0.2951f,  0.3244f },  // 92
            new float[] { 0.7117f, -0.6099f,  0.4946f, -0.4208f,  0.5476f,  0.1096f },  // 93
            new float[] {-0.2929f, -0.5726f,  0.5346f, -0.3827f,  0.4665f,  0.2465f },  // 94
            new float[] { 0.4889f, -0.5572f, -0.5718f, -0.6021f, -0.7150f,  0.2163f },  // 95
            new float[] {-0.7782f,  0.3491f,  0.5996f, -0.8389f, -0.5366f,  0.6516f },  // 96
            new float[] {-0.5847f,  0.8347f,  0.4226f,  0.1078f, -0.3910f,  0.6134f },  // 97
            new float[] { 0.8469f,  0.4121f, -0.0439f, -0.7476f,  0.9521f,  0.1571f },  // 98
            new float[] {-0.6803f, -0.5948f, -0.1376f, -0.1916f, -0.7065f,  0.7156f },  // 99
            new float[] { 0.2878f,  0.5086f, -0.5785f,  0.2019f,  0.4979f,  0.2980f },  // 100
            new float[] { 0.2764f,  0.1943f, -0.4090f,  0.4632f,  0.8906f,  0.2960f },  // 101
            new float[] {-0.8877f,  0.6705f, -0.6155f, -0.2098f, -0.3998f,  0.7107f },  // 102
            new float[] {-0.8398f,  0.8093f, -0.2597f,  0.0614f, -0.0118f,  0.6502f },  // 103
            new float[] {-0.8476f,  0.0158f, -0.4769f, -0.2859f, -0.7839f,  0.7715f },  // 104
            new float[] { 0.5751f, -0.7868f,  0.9714f, -0.6457f,  0.1448f,  0.1175f },  // 105
            new float[] { 0.4802f, -0.7001f,  0.1022f, -0.5668f,  0.5184f,  0.1090f },  // 106
            new float[] { 0.4458f, -0.6469f,  0.7239f, -0.9604f,  0.7205f,  0.0779f },  // 107
            new float[] { 0.5175f,  0.4339f,  0.9747f, -0.4438f, -0.9924f,  0.2879f },  // 108
            new float[] { 0.8678f,  0.7158f,  0.4577f,  0.0334f,  0.4139f,  0.1678f },  // 109
            new float[] { 0.5406f,  0.5012f,  0.2264f, -0.1963f,  0.3946f,  0.2088f },  // 110
            new float[] {-0.9938f,  0.5498f,  0.7928f, -0.5214f, -0.7585f,  0.7687f },  // 111
            new float[] { 0.7661f,  0.0863f, -0.4266f, -0.7233f, -0.4197f,  0.1466f },  // 112
            new float[] { 0.2277f, -0.3517f, -0.0853f, -0.1118f,  0.6563f,  0.1767f },  // 113
            new float[] { 0.3499f, -0.5570f, -0.0655f, -0.3705f,  0.2537f,  0.1632f },  // 114
            new float[] { 0.7547f, -0.1046f,  0.5689f, -0.0861f,  0.3125f,  0.1257f },  // 115
            new float[] { 0.8186f,  0.2110f,  0.5335f,  0.0094f, -0.0039f,  0.1391f },  // 116
            new float[] { 0.6858f, -0.8644f,  0.1465f,  0.8855f,  0.0357f,  0.1845f },  // 117
            new float[] {-0.4967f,  0.4015f,  0.0805f,  0.8977f,  0.2487f,  0.4663f },  // 118
            new float[] { 0.6760f, -0.9841f,  0.9787f, -0.8446f, -0.3557f,  0.1509f },  // 119
            new float[] {-0.1203f, -0.4885f,  0.6054f, -0.0443f, -0.7313f,  0.4854f },  // 120
            new float[] { 0.8557f,  0.7919f, -0.0169f,  0.7134f, -0.1628f,  0.2002f },  // 121
            new float[] { 0.0115f, -0.6209f,  0.9300f, -0.4116f, -0.7931f,  0.4052f },  // 122
            new float[] {-0.7114f, -0.9718f,  0.4319f,  0.1290f,  0.5892f,  0.3661f },  // 123
            new float[] { 0.3915f,  0.5557f, -0.1870f,  0.2955f, -0.6404f,  0.2954f },  // 124
            new float[] {-0.3564f, -0.6548f, -0.1827f, -0.5172f, -0.1862f,  0.4622f },  // 125
            new float[] { 0.2392f, -0.4959f,  0.5857f, -0.1341f, -0.2850f,  0.2470f },  // 126
            new float[] {-0.3394f,  0.3947f, -0.4627f,  0.6166f, -0.4094f,  0.5325f },  // 127
            new float[] { 0.7107f,  0.7768f, -0.6312f,  0.1707f,  0.7964f,  0.2757f },  // 128
            new float[] {-0.1078f,  0.8437f, -0.4420f,  0.2177f,  0.3649f,  0.4028f },  // 129
            new float[] {-0.3139f,  0.5595f, -0.6505f, -0.3161f, -0.7108f,  0.5546f },  // 130
            new float[] { 0.4335f,  0.3986f,  0.3770f, -0.4932f,  0.3847f,  0.1810f },  // 131
            new float[] {-0.2562f, -0.2894f, -0.8847f,  0.2633f,  0.4146f,  0.4036f },  // 132
            new float[] { 0.2272f,  0.2966f, -0.6601f, -0.7011f,  0.0284f,  0.2778f },  // 133
            new float[] {-0.0743f, -0.1421f, -0.0054f, -0.6770f, -0.3151f,  0.3597f },  // 134
            new float[] {-0.4762f,  0.6891f,  0.6007f, -0.1467f,  0.2140f,  0.4266f },  // 135
            new float[] {-0.4061f,  0.7193f,  0.3432f,  0.2669f, -0.7505f,  0.6147f },  // 136
            new float[] {-0.0588f,  0.9731f,  0.8966f,  0.2902f, -0.6966f,  0.4955f },  // 137
            new float[] {-0.0627f, -0.1439f,  0.1985f,  0.6999f,  0.5022f,  0.3077f },  // 138
            new float[] { 0.1587f,  0.8494f, -0.8705f,  0.9827f, -0.8940f,  0.4263f },  // 139
            new float[] {-0.7850f,  0.2473f, -0.9040f, -0.4308f, -0.8779f,  0.7199f },  // 140
            new float[] { 0.4070f,  0.3369f, -0.2428f, -0.6236f,  0.4940f,  0.2215f },  // 141
            new float[] {-0.0242f,  0.0513f, -0.9430f,  0.2885f, -0.2987f,  0.3947f },  // 142
            new float[] {-0.5416f, -0.1322f, -0.2351f, -0.0604f,  0.9590f,  0.3683f },  // 143
            new float[] { 0.1055f,  0.7783f, -0.2901f, -0.5090f,  0.8220f,  0.2984f },  // 144
            new float[] {-0.9129f,  0.9015f,  0.1128f, -0.2473f,  0.9901f,  0.4776f },  // 145
            new float[] {-0.9378f,  0.1424f, -0.6391f,  0.2619f,  0.9618f,  0.5368f },  // 146
            new float[] { 0.7498f, -0.0963f,  0.4169f,  0.5549f, -0.0103f,  0.1614f },  // 147
            new float[] {-0.2612f, -0.7156f,  0.4538f, -0.0460f, -0.1022f,  0.3717f },  // 148
            new float[] { 0.7720f,  0.0552f, -0.1818f, -0.4622f, -0.8560f,  0.1685f },  // 149
            new float[] {-0.4177f,  0.0070f,  0.9319f, -0.7812f,  0.3461f,  0.3052f },  // 150
            new float[] {-0.0001f,  0.5542f, -0.7128f, -0.8336f, -0.2016f,  0.3803f },  // 151
            new float[] { 0.5356f, -0.4194f, -0.5662f, -0.9666f, -0.2027f,  0.1776f },  // 152
            new float[] {-0.2378f,  0.3187f, -0.8582f, -0.6948f, -0.9668f,  0.5474f },  // 153
            new float[] {-0.1947f, -0.3579f,  0.1158f,  0.9869f,  0.6690f,  0.2992f },  // 154
            new float[] { 0.3992f,  0.8365f, -0.9205f, -0.8593f, -0.0520f,  0.3154f },  // 155
            new float[] {-0.0209f,  0.0793f,  0.7905f, -0.1067f,  0.7541f,  0.1864f },  // 156
            new float[] {-0.4928f, -0.4524f, -0.3433f,  0.0951f, -0.5597f,  0.6261f },  // 157
            new float[] {-0.8118f,  0.7404f, -0.5263f, -0.2280f,  0.1431f,  0.6349f },  // 158
            new float[] { 0.0516f, -0.8480f,  0.7483f,  0.9023f,  0.6250f,  0.1959f },  // 159
            new float[] {-0.3212f,  0.1093f,  0.9488f, -0.3766f,  0.3376f,  0.2735f },  // 160
            new float[] {-0.3481f,  0.5490f, -0.3484f,  0.7797f,  0.5034f,  0.4379f },  // 161
            new float[] {-0.5785f, -0.9170f, -0.3563f, -0.9258f,  0.3877f,  0.4121f },  // 162
            new float[] { 0.3407f, -0.1391f,  0.5356f,  0.0720f, -0.9203f,  0.3458f },  // 163
            new float[] {-0.3287f, -0.8954f,  0.2102f,  0.0241f,  0.2349f,  0.3247f },  // 164
            new float[] {-0.1353f,  0.6954f, -0.0919f, -0.9692f,  0.7461f,  0.3338f },  // 165
            new float[] { 0.9036f, -0.8982f, -0.5299f, -0.8733f, -0.1567f,  0.1187f },  // 166
            new float[] { 0.7277f, -0.8368f, -0.0538f, -0.7489f,  0.5458f,  0.0830f },  // 167
            new float[] { 0.9049f,  0.8878f,  0.2279f,  0.9470f, -0.3103f,  0.2194f },  // 168
            new float[] { 0.7957f, -0.1308f, -0.5284f,  0.8817f,  0.3684f,  0.2172f },  // 169
            new float[] { 0.4647f, -0.4931f,  0.2010f,  0.6292f, -0.8918f,  0.3371f },  // 170
            new float[] {-0.7390f,  0.6849f,  0.2367f,  0.0626f, -0.5034f,  0.7039f },  // 171
            new float[] {-0.1567f, -0.8711f,  0.7940f, -0.5932f,  0.6525f,  0.1710f },  // 172
            new float[] { 0.7635f, -0.0265f,  0.1969f,  0.0545f,  0.2496f,  0.1445f },  // 173
            new float[] { 0.7675f,  0.1354f, -0.7698f, -0.5460f,  0.1920f,  0.1728f },  // 174
            new float[] {-0.5211f, -0.7372f, -0.6763f,  0.6897f,  0.2044f,  0.5217f },  // 175
            new float[] { 0.1913f,  0.1980f,  0.2314f, -0.8816f,  0.5006f,  0.1998f },  // 176
            new float[] { 0.8964f,  0.0694f, -0.6149f,  0.5059f, -0.9854f,  0.1825f },  // 177
            new float[] { 0.1767f,  0.7104f,  0.2093f,  0.6452f,  0.7590f,  0.2832f },  // 178
            new float[] {-0.3580f, -0.7541f,  0.4426f, -0.1193f, -0.7465f,  0.5657f },  // 179
            new float[] {-0.5996f,  0.5766f, -0.9758f, -0.3933f, -0.9572f,  0.6800f },  // 180
            new float[] { 0.9950f,  0.1641f, -0.4132f,  0.8579f,  0.0142f,  0.2003f },  // 181
            new float[] {-0.4717f, -0.3894f, -0.2567f, -0.5111f,  0.1691f,  0.4266f },  // 182
            new float[] { 0.3917f, -0.8561f,  0.9422f,  0.5061f,  0.6123f,  0.1212f },  // 183
            new float[] {-0.0366f, -0.1087f,  0.3449f, -0.1025f,  0.4086f,  0.2475f },  // 184
            new float[] { 0.3633f,  0.3943f,  0.2372f, -0.6980f,  0.5216f,  0.1925f },  // 185
            new float[] {-0.5325f, -0.6466f, -0.2178f, -0.3589f,  0.6310f,  0.3568f },  // 186
            new float[] { 0.2271f,  0.5200f, -0.1447f, -0.8011f, -0.7699f,  0.3128f },  // 187
            new float[] { 0.6415f,  0.1993f,  0.3777f, -0.0178f, -0.8237f,  0.2181f },  // 188
            new float[] {-0.5298f, -0.0768f, -0.6028f, -0.9490f,  0.4588f,  0.4356f },  // 189
            new float[] { 0.6870f, -0.1431f,  0.7294f,  0.3141f,  0.1621f,  0.1632f },  // 190
            new float[] {-0.5985f,  0.0591f,  0.7889f, -0.3900f,  0.7419f,  0.2945f },  // 191
            new float[] { 0.3661f,  0.7984f, -0.8486f,  0.7572f, -0.6183f,  0.3449f },  // 192
            new float[] { 0.6995f,  0.3342f, -0.3113f, -0.6972f,  0.2707f,  0.1712f },  // 193
            new float[] { 0.2565f,  0.9126f,  0.1798f, -0.6043f, -0.1413f,  0.2893f },  // 194
            new float[] {-0.3265f,  0.9839f, -0.2395f,  0.9854f,  0.0376f,  0.4770f },  // 195
            new float[] { 0.2690f, -0.1722f,  0.9818f,  0.8599f, -0.7015f,  0.3954f },  // 196
            new float[] {-0.2102f, -0.0768f,  0.1219f,  0.5607f, -0.0256f,  0.3949f },  // 197
            new float[] { 0.8216f, -0.9555f,  0.6422f, -0.6231f,  0.3715f,  0.0801f },  // 198
            new float[] {-0.2896f,  0.9484f, -0.7545f, -0.6249f,  0.7789f,  0.4370f },  // 199
            new float[] {-0.9985f, -0.5448f, -0.7092f, -0.5931f,  0.7926f,  0.5402f }   // 200
        };
        return trainData;
    }
    public static float[][] TestDataSynth24()
    {
        float[][] testData = new float[][]
        {
           new float[] {  0.7462f,  0.4006f, -0.0590f,  0.6543f, -0.0083f,  0.1935f },  // 0
           new float[] {  0.8495f, -0.2260f, -0.0142f, -0.4911f,  0.7699f,  0.1078f },  // 1
           new float[] { -0.2335f, -0.4049f,  0.4352f, -0.6183f, -0.7636f,  0.5088f },  // 2
           new float[] {  0.1810f, -0.5142f,  0.2465f,  0.2767f, -0.3449f,  0.3136f },  // 3
           new float[] { -0.8650f,  0.7611f, -0.0801f,  0.5277f, -0.4922f,  0.7140f },  // 4
           new float[] { -0.2358f, -0.7466f, -0.5115f, -0.8413f, -0.3943f,  0.4533f },  // 5
           new float[] {  0.4834f,  0.2300f,  0.3448f, -0.9832f,  0.3568f,  0.1360f },  // 6
           new float[] { -0.6502f, -0.6300f,  0.6885f,  0.9652f,  0.8275f,  0.3046f },  // 7
           new float[] { -0.3053f,  0.5604f,  0.0929f,  0.6329f, -0.0325f,  0.4756f },  // 8
           new float[] { -0.7995f,  0.0740f, -0.2680f,  0.2086f,  0.9176f,  0.4565f },  // 9
           new float[] { -0.2144f, -0.2141f,  0.5813f,  0.2902f, -0.2122f,  0.4119f }, // 10
           new float[] { -0.7278f, -0.0987f, -0.3312f, -0.5641f,  0.8515f,  0.4438f }, // 11
           new float[] {  0.3793f,  0.1976f,  0.4933f,  0.0839f,  0.4011f,  0.1905f }, // 12
           new float[] { -0.8568f,  0.9573f, -0.5272f,  0.3212f, -0.8207f,  0.7415f }, // 13
           new float[] { -0.5785f,  0.0056f, -0.7901f, -0.2223f,  0.0760f,  0.5551f }, // 14
           new float[] {  0.0735f, -0.2188f,  0.3925f,  0.3570f,  0.3746f,  0.2191f }, // 15
           new float[] {  0.1230f, -0.2838f,  0.2262f,  0.8715f,  0.1938f,  0.2878f }, // 16
           new float[] {  0.4792f, -0.9248f,  0.5295f,  0.0366f, -0.9894f,  0.3149f }, // 17
           new float[] { -0.4456f,  0.0697f,  0.5359f, -0.8938f,  0.0981f,  0.3879f }, // 18
           new float[] {  0.8629f, -0.8505f, -0.4464f,  0.8385f,  0.5300f,  0.1769f }, // 19
           new float[] {  0.1995f,  0.6659f,  0.7921f,  0.9454f,  0.9970f,  0.2330f }, // 20
           new float[] { -0.0249f, -0.3066f, -0.2927f, -0.4923f,  0.8220f,  0.2437f }, // 21
           new float[] {  0.4513f, -0.9481f, -0.0770f, -0.4374f, -0.9421f,  0.2879f }, // 22
           new float[] { -0.3405f,  0.5931f, -0.3507f, -0.3842f,  0.8562f,  0.3987f }, // 23
           new float[] {  0.9538f,  0.0471f,  0.9039f,  0.7760f,  0.0361f,  0.1706f }, // 24
           new float[] { -0.0887f,  0.2104f,  0.9808f,  0.5478f, -0.3314f,  0.4128f }, // 25
           new float[] { -0.8220f, -0.6302f,  0.0537f, -0.1658f,  0.6013f,  0.4306f }, // 26
           new float[] { -0.4123f, -0.2880f,  0.9074f, -0.0461f, -0.4435f,  0.5144f }, // 27
           new float[] {  0.0060f,  0.2867f, -0.7775f,  0.5161f,  0.7039f,  0.3599f }, // 28
           new float[] { -0.7968f, -0.5484f,  0.9426f, -0.4308f,  0.8148f,  0.2979f }, // 29
           new float[] {  0.7811f,  0.8450f, -0.6877f,  0.7594f,  0.2640f,  0.2362f }, // 30
           new float[] { -0.6802f, -0.1113f, -0.8325f, -0.6694f, -0.6056f,  0.6544f }, // 31
           new float[] {  0.3821f,  0.1476f,  0.7466f, -0.5107f,  0.2592f,  0.1648f }, // 32
           new float[] {  0.7265f,  0.9683f, -0.9803f, -0.4943f, -0.5523f,  0.2454f }, // 33
           new float[] { -0.9049f, -0.9797f, -0.0196f, -0.9090f, -0.4433f,  0.6447f }, // 34
           new float[] { -0.4607f,  0.1811f, -0.2389f,  0.4050f, -0.0078f,  0.5229f }, // 35
           new float[] {  0.2664f, -0.2932f, -0.4259f, -0.7336f,  0.8742f,  0.1834f }, // 36
           new float[] { -0.4507f,  0.1029f, -0.6294f, -0.1158f, -0.6294f,  0.6081f }, // 37
           new float[] {  0.8948f, -0.0124f,  0.9278f,  0.2899f, -0.0314f,  0.1534f }, // 38
           new float[] { -0.1323f, -0.8813f, -0.0146f, -0.0697f,  0.6135f,  0.2386f }  // 39
        };
        return testData;
    }

    public static float[][] TrainDataPeople24()
    {
        float[][] trainData = new float[][]
        {
            new float[] { 1, 0.24f, 1, 0, 0, 0.2950f, 0, 0, 1 },  // 1
            new float[] {-1, 0.39f, 0, 0, 1, 0.5120f, 0, 1, 0 },  // 2
            new float[] { 1, 0.63f, 0, 1, 0, 0.7580f, 1, 0, 0 },  // 3
            new float[] {-1, 0.36f, 1, 0, 0, 0.4450f, 0, 1, 0 },  // 4
            new float[] { 1, 0.27f, 0, 1, 0, 0.2860f, 0, 0, 1 },  // 5
            new float[] { 1, 0.50f, 0, 1, 0, 0.5650f, 0, 1, 0 },  // 6
            new float[] { 1, 0.50f, 0, 0, 1, 0.5500f, 0, 1, 0 },  // 7
            new float[] {-1, 0.19f, 0, 0, 1, 0.3270f, 1, 0, 0 },  // 8
            new float[] { 1, 0.22f, 0, 1, 0, 0.2770f, 0, 1, 0 },  // 9
            new float[] {-1, 0.39f, 0, 0, 1, 0.4710f, 0, 0, 1 },  // 10
            new float[] { 1, 0.34f, 1, 0, 0, 0.3940f, 0, 1, 0 },  // 11
            new float[] {-1, 0.22f, 1, 0, 0, 0.3350f, 1, 0, 0 },  // 12
            new float[] { 1, 0.35f, 0, 0, 1, 0.3520f, 0, 0, 1 },  // 13
            new float[] {-1, 0.33f, 0, 1, 0, 0.4640f, 0, 1, 0 },  // 14
            new float[] { 1, 0.45f, 0, 1, 0, 0.5410f, 0, 1, 0 },  // 15
            new float[] { 1, 0.42f, 0, 1, 0, 0.5070f, 0, 1, 0 },  // 16
            new float[] {-1, 0.33f, 0, 1, 0, 0.4680f, 0, 1, 0 },  // 17
            new float[] { 1, 0.25f, 0, 0, 1, 0.3000f, 0, 1, 0 },  // 18
            new float[] {-1, 0.31f, 0, 1, 0, 0.4640f, 1, 0, 0 },  // 19
            new float[] { 1, 0.27f, 1, 0, 0, 0.3250f, 0, 0, 1 },  // 20
            new float[] { 1, 0.48f, 1, 0, 0, 0.5400f, 0, 1, 0 },  // 21
            new float[] {-1, 0.64f, 0, 1, 0, 0.7130f, 0, 0, 1 },  // 22
            new float[] { 1, 0.61f, 0, 1, 0, 0.7240f, 1, 0, 0 },  // 23
            new float[] { 1, 0.54f, 0, 0, 1, 0.6100f, 1, 0, 0 },  // 24
            new float[] { 1, 0.29f, 1, 0, 0, 0.3630f, 1, 0, 0 },  // 25
            new float[] { 1, 0.50f, 0, 0, 1, 0.5500f, 0, 1, 0 },  // 26
            new float[] { 1, 0.55f, 0, 0, 1, 0.6250f, 1, 0, 0 },  // 27
            new float[] { 1, 0.40f, 1, 0, 0, 0.5240f, 1, 0, 0 },  // 28
            new float[] { 1, 0.22f, 1, 0, 0, 0.2360f, 0, 0, 1 },  // 29
            new float[] { 1, 0.68f, 0, 1, 0, 0.7840f, 1, 0, 0 },  // 30
            new float[] {-1, 0.60f, 1, 0, 0, 0.7170f, 0, 0, 1 },  // 31
            new float[] {-1, 0.34f, 0, 0, 1, 0.4650f, 0, 1, 0 },  // 32
            new float[] {-1, 0.25f, 0, 0, 1, 0.3710f, 1, 0, 0 },  // 33
            new float[] {-1, 0.31f, 0, 1, 0, 0.4890f, 0, 1, 0 },  // 34
            new float[] { 1, 0.43f, 0, 0, 1, 0.4800f, 0, 1, 0 },  // 35
            new float[] { 1, 0.58f, 0, 1, 0, 0.6540f, 0, 0, 1 },  // 36
            new float[] {-1, 0.55f, 0, 1, 0, 0.6070f, 0, 0, 1 },  // 37
            new float[] {-1, 0.43f, 0, 1, 0, 0.5110f, 0, 1, 0 },  // 38
            new float[] {-1, 0.43f, 0, 0, 1, 0.5320f, 0, 1, 0 },  // 39
            new float[] {-1, 0.21f, 1, 0, 0, 0.3720f, 1, 0, 0 },  // 40
            new float[] { 1, 0.55f, 0, 0, 1, 0.6460f, 1, 0, 0 },  // 41
            new float[] { 1, 0.64f, 0, 1, 0, 0.7480f, 1, 0, 0 },  // 42
            new float[] {-1, 0.41f, 1, 0, 0, 0.5880f, 0, 1, 0 },  // 43
            new float[] { 1, 0.64f, 0, 0, 1, 0.7270f, 1, 0, 0 },  // 44
            new float[] {-1, 0.56f, 0, 0, 1, 0.6660f, 0, 0, 1 },  // 45
            new float[] { 1, 0.31f, 0, 0, 1, 0.3600f, 0, 1, 0 },  // 46
            new float[] {-1, 0.65f, 0, 0, 1, 0.7010f, 0, 0, 1 },  // 47
            new float[] { 1, 0.55f, 0, 0, 1, 0.6430f, 1, 0, 0 },  // 48
            new float[] {-1, 0.25f, 1, 0, 0, 0.4030f, 1, 0, 0 },  // 49
            new float[] { 1, 0.46f, 0, 0, 1, 0.5100f, 0, 1, 0 },  // 50
            new float[] {-1, 0.36f, 1, 0, 0, 0.5350f, 1, 0, 0 },  // 51
            new float[] { 1, 0.52f, 0, 1, 0, 0.5810f, 0, 1, 0 },  // 52
            new float[] { 1, 0.61f, 0, 0, 1, 0.6790f, 1, 0, 0 },  // 53
            new float[] { 1, 0.57f, 0, 0, 1, 0.6570f, 1, 0, 0 },  // 54
            new float[] {-1, 0.46f, 0, 1, 0, 0.5260f, 0, 1, 0 },  // 55
            new float[] {-1, 0.62f, 1, 0, 0, 0.6680f, 0, 0, 1 },  // 56
            new float[] { 1, 0.55f, 0, 0, 1, 0.6270f, 1, 0, 0 },  // 57
            new float[] {-1, 0.22f, 0, 0, 1, 0.2770f, 0, 1, 0 },  // 58
            new float[] {-1, 0.50f, 1, 0, 0, 0.6290f, 1, 0, 0 },  // 59
            new float[] {-1, 0.32f, 0, 1, 0, 0.4180f, 0, 1, 0 },  // 60
            new float[] {-1, 0.21f, 0, 0, 1, 0.3560f, 1, 0, 0 },  // 61
            new float[] { 1, 0.44f, 0, 1, 0, 0.5200f, 0, 1, 0 },  // 62
            new float[] { 1, 0.46f, 0, 1, 0, 0.5170f, 0, 1, 0 },  // 63
            new float[] { 1, 0.62f, 0, 1, 0, 0.6970f, 1, 0, 0 },  // 64
            new float[] { 1, 0.57f, 0, 1, 0, 0.6640f, 1, 0, 0 },  // 65
            new float[] {-1, 0.67f, 0, 0, 1, 0.7580f, 0, 0, 1 },  // 66
            new float[] { 1, 0.29f, 1, 0, 0, 0.3430f, 0, 0, 1 },  // 67
            new float[] { 1, 0.53f, 1, 0, 0, 0.6010f, 1, 0, 0 },  // 68
            new float[] {-1, 0.44f, 1, 0, 0, 0.5480f, 0, 1, 0 },  // 69
            new float[] { 1, 0.46f, 0, 1, 0, 0.5230f, 0, 1, 0 },  // 70
            new float[] {-1, 0.20f, 0, 1, 0, 0.3010f, 0, 1, 0 },  // 71
            new float[] {-1, 0.38f, 1, 0, 0, 0.5350f, 0, 1, 0 },  // 72
            new float[] { 1, 0.50f, 0, 1, 0, 0.5860f, 0, 1, 0 },  // 73
            new float[] { 1, 0.33f, 0, 1, 0, 0.4250f, 0, 1, 0 },  // 74
            new float[] {-1, 0.33f, 0, 1, 0, 0.3930f, 0, 1, 0 },  // 75
            new float[] { 1, 0.26f, 0, 1, 0, 0.4040f, 1, 0, 0 },  // 76
            new float[] { 1, 0.58f, 1, 0, 0, 0.7070f, 1, 0, 0 },  // 77
            new float[] { 1, 0.43f, 0, 0, 1, 0.4800f, 0, 1, 0 },  // 78
            new float[] {-1, 0.46f, 1, 0, 0, 0.6440f, 1, 0, 0 },  // 79
            new float[] { 1, 0.60f, 1, 0, 0, 0.7170f, 1, 0, 0 },  // 80
            new float[] {-1, 0.42f, 1, 0, 0, 0.4890f, 0, 1, 0 },  // 81
            new float[] {-1, 0.56f, 0, 0, 1, 0.5640f, 0, 0, 1 },  // 82
            new float[] {-1, 0.62f, 0, 1, 0, 0.6630f, 0, 0, 1 },  // 83
            new float[] {-1, 0.50f, 1, 0, 0, 0.6480f, 0, 1, 0 },  // 84
            new float[] { 1, 0.47f, 0, 0, 1, 0.5200f, 0, 1, 0 },  // 85
            new float[] {-1, 0.67f, 0, 1, 0, 0.8040f, 0, 0, 1 },  // 86
            new float[] {-1, 0.40f, 0, 0, 1, 0.5040f, 0, 1, 0 },  // 87
            new float[] { 1, 0.42f, 0, 1, 0, 0.4840f, 0, 1, 0 },  // 88
            new float[] { 1, 0.64f, 1, 0, 0, 0.7200f, 1, 0, 0 },  // 89
            new float[] {-1, 0.47f, 1, 0, 0, 0.5870f, 0, 0, 1 },  // 90
            new float[] { 1, 0.45f, 0, 1, 0, 0.5280f, 0, 1, 0 },  // 91
            new float[] {-1, 0.25f, 0, 0, 1, 0.4090f, 1, 0, 0 },  // 92
            new float[] { 1, 0.38f, 1, 0, 0, 0.4840f, 1, 0, 0 },  // 93
            new float[] { 1, 0.55f, 0, 0, 1, 0.6000f, 0, 1, 0 },  // 94
            new float[] {-1, 0.44f, 1, 0, 0, 0.6060f, 0, 1, 0 },  // 95
            new float[] { 1, 0.33f, 1, 0, 0, 0.4100f, 0, 1, 0 },  // 96
            new float[] { 1, 0.34f, 0, 0, 1, 0.3900f, 0, 1, 0 },  // 97
            new float[] { 1, 0.27f, 0, 1, 0, 0.3370f, 0, 0, 1 },  // 98
            new float[] { 1, 0.32f, 0, 1, 0, 0.4070f, 0, 1, 0 },  // 99
            new float[] { 1, 0.42f, 0, 0, 1, 0.4700f, 0, 1, 0 },  // 100
            new float[] {-1, 0.24f, 0, 0, 1, 0.4030f, 1, 0, 0 },  // 101
            new float[] { 1, 0.42f, 0, 1, 0, 0.5030f, 0, 1, 0 },  // 102
            new float[] { 1, 0.25f, 0, 0, 1, 0.2800f, 0, 0, 1 },  // 103
            new float[] { 1, 0.51f, 0, 1, 0, 0.5800f, 0, 1, 0 },  // 104
            new float[] {-1, 0.55f, 0, 1, 0, 0.6350f, 0, 0, 1 },  // 105
            new float[] { 1, 0.44f, 1, 0, 0, 0.4780f, 0, 0, 1 },  // 106
            new float[] {-1, 0.18f, 1, 0, 0, 0.3980f, 1, 0, 0 },  // 107
            new float[] {-1, 0.67f, 0, 1, 0, 0.7160f, 0, 0, 1 },  // 108
            new float[] { 1, 0.45f, 0, 0, 1, 0.5000f, 0, 1, 0 },  // 109
            new float[] { 1, 0.48f, 1, 0, 0, 0.5580f, 0, 1, 0 },  // 110
            new float[] {-1, 0.25f, 0, 1, 0, 0.3900f, 0, 1, 0 },  // 111
            new float[] {-1, 0.67f, 1, 0, 0, 0.7830f, 0, 1, 0 },  // 112
            new float[] { 1, 0.37f, 0, 0, 1, 0.4200f, 0, 1, 0 },  // 113
            new float[] {-1, 0.32f, 1, 0, 0, 0.4270f, 0, 1, 0 },  // 114
            new float[] { 1, 0.48f, 1, 0, 0, 0.5700f, 0, 1, 0 },  // 115
            new float[] {-1, 0.66f, 0, 0, 1, 0.7500f, 0, 0, 1 },  // 116
            new float[] { 1, 0.61f, 1, 0, 0, 0.7000f, 1, 0, 0 },  // 117
            new float[] {-1, 0.58f, 0, 0, 1, 0.6890f, 0, 1, 0 },  // 118
            new float[] { 1, 0.19f, 1, 0, 0, 0.2400f, 0, 0, 1 },  // 119
            new float[] { 1, 0.38f, 0, 0, 1, 0.4300f, 0, 1, 0 },  // 120
            new float[] {-1, 0.27f, 1, 0, 0, 0.3640f, 0, 1, 0 },  // 121
            new float[] { 1, 0.42f, 1, 0, 0, 0.4800f, 0, 1, 0 },  // 122
            new float[] { 1, 0.60f, 1, 0, 0, 0.7130f, 1, 0, 0 },  // 123
            new float[] {-1, 0.27f, 0, 0, 1, 0.3480f, 1, 0, 0 },  // 124
            new float[] { 1, 0.29f, 0, 1, 0, 0.3710f, 1, 0, 0 },  // 125
            new float[] {-1, 0.43f, 1, 0, 0, 0.5670f, 0, 1, 0 },  // 126
            new float[] { 1, 0.48f, 1, 0, 0, 0.5670f, 0, 1, 0 },  // 127
            new float[] { 1, 0.27f, 0, 0, 1, 0.2940f, 0, 0, 1 },  // 128
            new float[] {-1, 0.44f, 1, 0, 0, 0.5520f, 1, 0, 0 },  // 129
            new float[] { 1, 0.23f, 0, 1, 0, 0.2630f, 0, 0, 1 },  // 130
            new float[] {-1, 0.36f, 0, 1, 0, 0.5300f, 0, 0, 1 },  // 131
            new float[] { 1, 0.64f, 0, 0, 1, 0.7250f, 1, 0, 0 },  // 132
            new float[] { 1, 0.29f, 0, 0, 1, 0.3000f, 0, 0, 1 },  // 133
            new float[] {-1, 0.33f, 1, 0, 0, 0.4930f, 0, 1, 0 },  // 134
            new float[] {-1, 0.66f, 0, 1, 0, 0.7500f, 0, 0, 1 },  // 135
            new float[] {-1, 0.21f, 0, 0, 1, 0.3430f, 1, 0, 0 },  // 136
            new float[] { 1, 0.27f, 1, 0, 0, 0.3270f, 0, 0, 1 },  // 137
            new float[] { 1, 0.29f, 1, 0, 0, 0.3180f, 0, 0, 1 },  // 138
            new float[] {-1, 0.31f, 1, 0, 0, 0.4860f, 0, 1, 0 },  // 139
            new float[] { 1, 0.36f, 0, 0, 1, 0.4100f, 0, 1, 0 },  // 140
            new float[] { 1, 0.49f, 0, 1, 0, 0.5570f, 0, 1, 0 },  // 141
            new float[] {-1, 0.28f, 1, 0, 0, 0.3840f, 1, 0, 0 },  // 142
            new float[] {-1, 0.43f, 0, 0, 1, 0.5660f, 0, 1, 0 },  // 143
            new float[] {-1, 0.46f, 0, 1, 0, 0.5880f, 0, 1, 0 },  // 144
            new float[] { 1, 0.57f, 1, 0, 0, 0.6980f, 1, 0, 0 },  // 145
            new float[] {-1, 0.52f, 0, 0, 1, 0.5940f, 0, 1, 0 },  // 146
            new float[] {-1, 0.31f, 0, 0, 1, 0.4350f, 0, 1, 0 },  // 147
            new float[] {-1, 0.55f, 1, 0, 0, 0.6200f, 0, 0, 1 },  // 148
            new float[] { 1, 0.50f, 1, 0, 0, 0.5640f, 0, 1, 0 },  // 149
            new float[] { 1, 0.48f, 0, 1, 0, 0.5590f, 0, 1, 0 },  // 150
            new float[] {-1, 0.22f, 0, 0, 1, 0.3450f, 1, 0, 0 },  // 151
            new float[] { 1, 0.59f, 0, 0, 1, 0.6670f, 1, 0, 0 },  // 152
            new float[] { 1, 0.34f, 1, 0, 0, 0.4280f, 0, 0, 1 },  // 153
            new float[] {-1, 0.64f, 1, 0, 0, 0.7720f, 0, 0, 1 },  // 154
            new float[] { 1, 0.29f, 0, 0, 1, 0.3350f, 0, 0, 1 },  // 155
            new float[] {-1, 0.34f, 0, 1, 0, 0.4320f, 0, 1, 0 },  // 156
            new float[] {-1, 0.61f, 1, 0, 0, 0.7500f, 0, 0, 1 },  // 157
            new float[] { 1, 0.64f, 0, 0, 1, 0.7110f, 1, 0, 0 },  // 158
            new float[] {-1, 0.29f, 1, 0, 0, 0.4130f, 1, 0, 0 },  // 159
            new float[] { 1, 0.63f, 0, 1, 0, 0.7060f, 1, 0, 0 },  // 160
            new float[] {-1, 0.29f, 0, 1, 0, 0.4000f, 1, 0, 0 },  // 161
            new float[] {-1, 0.51f, 1, 0, 0, 0.6270f, 0, 1, 0 },  // 162
            new float[] {-1, 0.24f, 0, 0, 1, 0.3770f, 1, 0, 0 },  // 163
            new float[] { 1, 0.48f, 0, 1, 0, 0.5750f, 0, 1, 0 },  // 164
            new float[] { 1, 0.18f, 1, 0, 0, 0.2740f, 1, 0, 0 },  // 165
            new float[] { 1, 0.18f, 1, 0, 0, 0.2030f, 0, 0, 1 },  // 166
            new float[] { 1, 0.33f, 0, 1, 0, 0.3820f, 0, 0, 1 },  // 167
            new float[] {-1, 0.20f, 0, 0, 1, 0.3480f, 1, 0, 0 },  // 168
            new float[] { 1, 0.29f, 0, 0, 1, 0.3300f, 0, 0, 1 },  // 169
            new float[] {-1, 0.44f, 0, 0, 1, 0.6300f, 1, 0, 0 },  // 170
            new float[] {-1, 0.65f, 0, 0, 1, 0.8180f, 1, 0, 0 },  // 171
            new float[] {-1, 0.56f, 1, 0, 0, 0.6370f, 0, 0, 1 },  // 172
            new float[] {-1, 0.52f, 0, 0, 1, 0.5840f, 0, 1, 0 },  // 173
            new float[] {-1, 0.29f, 0, 1, 0, 0.4860f, 1, 0, 0 },  // 174
            new float[] {-1, 0.47f, 0, 1, 0, 0.5890f, 0, 1, 0 },  // 175
            new float[] { 1, 0.68f, 1, 0, 0, 0.7260f, 0, 0, 1 },  // 176
            new float[] { 1, 0.31f, 0, 0, 1, 0.3600f, 0, 1, 0 },  // 177
            new float[] { 1, 0.61f, 0, 1, 0, 0.6250f, 0, 0, 1 },  // 178
            new float[] { 1, 0.19f, 0, 1, 0, 0.2150f, 0, 0, 1 },  // 179
            new float[] { 1, 0.38f, 0, 0, 1, 0.4300f, 0, 1, 0 },  // 180
            new float[] {-1, 0.26f, 1, 0, 0, 0.4230f, 1, 0, 0 },  // 181
            new float[] { 1, 0.61f, 0, 1, 0, 0.6740f, 1, 0, 0 },  // 182
            new float[] { 1, 0.40f, 1, 0, 0, 0.4650f, 0, 1, 0 },  // 183
            new float[] {-1, 0.49f, 1, 0, 0, 0.6520f, 0, 1, 0 },  // 184
            new float[] { 1, 0.56f, 1, 0, 0, 0.6750f, 1, 0, 0 },  // 185
            new float[] {-1, 0.48f, 0, 1, 0, 0.6600f, 0, 1, 0 },  // 186
            new float[] { 1, 0.52f, 1, 0, 0, 0.5630f, 0, 0, 1 },  // 187
            new float[] {-1, 0.18f, 1, 0, 0, 0.2980f, 1, 0, 0 },  // 188
            new float[] {-1, 0.56f, 0, 0, 1, 0.5930f, 0, 0, 1 },  // 189
            new float[] {-1, 0.52f, 0, 1, 0, 0.6440f, 0, 1, 0 },  // 190
            new float[] {-1, 0.18f, 0, 1, 0, 0.2860f, 0, 1, 0 },  // 191
            new float[] {-1, 0.58f, 1, 0, 0, 0.6620f, 0, 0, 1 },  // 192
            new float[] {-1, 0.39f, 0, 1, 0, 0.5510f, 0, 1, 0 },  // 193
            new float[] {-1, 0.46f, 1, 0, 0, 0.6290f, 0, 1, 0 },  // 194
            new float[] {-1, 0.40f, 0, 1, 0, 0.4620f, 0, 1, 0 },  // 195
            new float[] {-1, 0.60f, 1, 0, 0, 0.7270f, 0, 0, 1 },  // 196
            new float[] { 1, 0.36f, 0, 1, 0, 0.4070f, 0, 0, 1 },  // 197
            new float[] { 1, 0.44f, 1, 0, 0, 0.5230f, 0, 1, 0 },  // 198
            new float[] { 1, 0.28f, 1, 0, 0, 0.3130f, 0, 0, 1 },  // 199
            new float[] { 1, 0.54f, 0, 0, 1, 0.6260f, 1, 0, 0 }   // 200
        };
        return trainData;
    }
    public static float[][] TestDataPeople24()
    {
        float[][] testData = new float[][]
        {
           new float[] { -1, 0.51f, 1, 0, 0, 0.6120f, 0, 1, 0 },  // 0
           new float[] { -1, 0.32f, 0, 1, 0, 0.4610f, 0, 1, 0 },  // 1
           new float[] {  1, 0.55f, 1, 0, 0, 0.6270f, 1, 0, 0 },  // 2
           new float[] {  1, 0.25f, 0, 0, 0, 0.2620f, 0, 0, 1 },  // 3
           new float[] {  1, 0.33f, 0, 0, 0, 0.3730f, 0, 0, 1 },  // 4
           new float[] { -1, 0.29f, 0, 1, 0, 0.4620f, 1, 0, 0 },  // 5
           new float[] {  1, 0.65f, 1, 0, 0, 0.7270f, 1, 0, 0 },  // 6
           new float[] { -1, 0.43f, 0, 1, 0, 0.5140f, 0, 1, 0 },  // 7
           new float[] { -1, 0.54f, 0, 1, 0, 0.6480f, 0, 0, 1 },  // 8
           new float[] {  1, 0.61f, 0, 1, 0, 0.7270f, 1, 0, 0 },  // 9
           new float[] {  1, 0.52f, 0, 1, 0, 0.6360f, 1, 0, 0 }, // 10
           new float[] {  1, 0.30f, 0, 1, 0, 0.3350f, 0, 0, 1 }, // 11
           new float[] {  1, 0.29f, 1, 0, 0, 0.3140f, 0, 0, 1 }, // 12
           new float[] { -1, 0.47f, 0, 0, 0, 0.5940f, 0, 1, 0 }, // 13
           new float[] {  1, 0.39f, 0, 1, 0, 0.4780f, 0, 1, 0 }, // 14
           new float[] {  1, 0.47f, 0, 0, 0, 0.5200f, 0, 1, 0 }, // 15
           new float[] { -1, 0.49f, 1, 0, 0, 0.5860f, 0, 1, 0 }, // 16
           new float[] { -1, 0.63f, 0, 0, 0, 0.6740f, 0, 0, 1 }, // 17
           new float[] { -1, 0.30f, 1, 0, 0, 0.3920f, 1, 0, 0 }, // 18
           new float[] { -1, 0.61f, 0, 0, 0, 0.6960f, 0, 0, 1 }, // 19
           new float[] { -1, 0.47f, 0, 0, 0, 0.5870f, 0, 1, 0 }, // 20
           new float[] {  1, 0.30f, 0, 0, 0, 0.3450f, 0, 0, 1 }, // 21
           new float[] { -1, 0.51f, 0, 0, 0, 0.5800f, 0, 1, 0 }, // 22
           new float[] { -1, 0.24f, 1, 0, 0, 0.3880f, 0, 1, 0 }, // 23
           new float[] { -1, 0.49f, 1, 0, 0, 0.6450f, 0, 1, 0 }, // 24
           new float[] {  1, 0.66f, 0, 0, 0, 0.7450f, 1, 0, 0 }, // 25
           new float[] { -1, 0.65f, 1, 0, 0, 0.7690f, 1, 0, 0 }, // 26
           new float[] { -1, 0.46f, 0, 1, 0, 0.5800f, 1, 0, 0 }, // 27
           new float[] { -1, 0.45f, 0, 0, 0, 0.5180f, 0, 1, 0 }, // 28
           new float[] { -1, 0.47f, 1, 0, 0, 0.6360f, 1, 0, 0 }, // 29
           new float[] { -1, 0.29f, 1, 0, 0, 0.4480f, 1, 0, 0 }, // 30
           new float[] { -1, 0.57f, 0, 0, 0, 0.6930f, 0, 0, 1 }, // 31
           new float[] { -1, 0.20f, 1, 0, 0, 0.2870f, 0, 0, 1 }, // 32
           new float[] { -1, 0.35f, 1, 0, 0, 0.4340f, 0, 1, 0 }, // 33
           new float[] { -1, 0.61f, 0, 0, 0, 0.6700f, 0, 0, 1 }, // 34
           new float[] { -1, 0.31f, 0, 0, 0, 0.3730f, 0, 1, 0 }, // 35
           new float[] {  1, 0.18f, 1, 0, 0, 0.2080f, 0, 0, 1 }, // 36
           new float[] {  1, 0.26f, 0, 0, 0, 0.2920f, 0, 0, 1 }, // 37
           new float[] { -1, 0.28f, 1, 0, 0, 0.3640f, 0, 0, 1 }, // 38
           new float[] { -1, 0.59f, 0, 0, 0, 0.6940f, 0, 0, 1 }  // 39
        };
        return testData;
    }
}
