# Neural Network Regression 2025 in C#: Predicting Income

This is my latest neural network implementation for regression problems, featuring a modern architecture that delivers highly accurate predictions. While I may not be the best at explaining technical details, tools like ChatGPT can help clarify the output and code. My current favorite LLM is [DeepSeek](https://chat.deepseek.com/). It’s not necessarily the most advanced model, but I appreciate its balanced approach compared to other major models. Let’s dive into the demo to see how it works.

---

## NN25 Demo with New Score Function  

<p align="center">
  <img src="https://github.com/grensen/neural_regression25/blob/main/nn25_regression.png">
</p>

[Demo Code](https://github.com/grensen/neural_regression25/blob/main/nn25_regression.cs)

---

The demo uses a **neural network** to predict **income** based on **8 input features** from the **People dataset**. The dataset includes **200 training samples** and **40 testing samples**. The neural network, with an **8-100-100-1 architecture**, is trained using **Stochastic Gradient Descent (SGD)** and optimized with hyperparameters like learning rate and momentum.

The model introduces a **composite score** where **lower values indicate better performance**. This score combines **RMSE (Root Mean Squared Error)**, **MAE (Mean Absolute Error)**, and **accuracy within thresholds (10%, 5%, 1%)**. The formula for the score is:  
**Score = ((RMSE + MAE) / 2) * (1 - (10% + 2 * 5% + 3 * 1%) / 6)**  

- **RMSE**: Measures the average magnitude of the prediction errors (lower is better).  
- **MAE**: Measures the average absolute difference between predictions and targets (lower is better).  
- **Accuracy within thresholds**: The percentage of predictions that fall within 1%, 5%, and 10% of the target values (higher is better).  

This score provides a single metric to evaluate the model's performance, balancing both error and accuracy.  

### Results  
The model achieves strong results:  
- **Training Score:** 0.955  
- **Testing Score:** 0.826  

Interestingly and unusually, the **test score is better (lower) than the training score**. This can happen due to:  
1. **Regularization effects** (e.g., weight decay, momentum) preventing overfitting to the training data.  
2. The **test set** being easier to predict due to its specific distribution or lower noise.  
3. Random variability in smaller datasets, where the test set aligns well with the model’s predictions.  

---

## Machine Learning Models 2025

<p align="center">
  <img src="https://github.com/grensen/neural_regression25/blob/main/bench_regression25.png">
</p>

The next figure compares the **neural network** with **10 other models** on **3 datasets**. While not the best performer, the neural network ranked **4th overall**, consistently placing in the top tier across all datasets. Only more advanced deep learning systems, typically requiring greater effort, outperformed it. This underscores the neural network’s **strong performance** and **practicality**.

---

## The Neural Network: How It Works

### 🧠 Flexible Architecture with `net`
The neural network is defined by the **`net` array** (`int[]`), where each value represents the number of neurons in a layer. For example:
- `[8, 1]`: A **linear regression model** (8 inputs → 1 output).
- `[8, 100, 1]`: A **single hidden layer neural network** (8 inputs → 100 neurons → 1 output).
- `[8, 100, 100, 1]`: A **two hidden layer neural network** (8 inputs → 100 neurons → 100 neurons → 1 output).

This design allows you to experiment with **any number of layers and neurons** without rewriting the core logic. Whether you’re building a simple linear model or a deep neural network, the same codebase adapts seamlessly.

---

### ⚡ ReLU Pre-Activation: Speed and Efficiency
The network uses **ReLU pre-activation**, meaning the activation function (ReLU) is applied **before** the weighted sum. This approach:
- **Speeds up computation**: ReLU is applied only to non-zero values, skipping unnecessary calculations.
- **Improves gradient flow**: By focusing on positive activations, it avoids the vanishing gradient problem common in deep networks.

For example, during forward propagation:
~~~cs
if (n > 0) // Pre-ReLU
    for (int r = 0; r < weights[jl].Length; r++)
        neurons[k + r] += weights[jl][r] * n;
~~~

---

### 🔄 Memory Reuse: Gradients and Neurons
One of the coolest optimizations in this implementation is **reusing the `neurons` array** for gradient computation during backpropagation. Here’s why this matters:
- **Efficiency**: Instead of allocating new memory for gradients, we reuse the existing `neurons` array. This reduces memory overhead and improves performance.
- **Propagation of Gradients**: Each input neuron’s gradient is computed once and reused across all connected neurons in the next layer. This avoids redundant calculations and speeds up training.

~~~cs
neurons[jl] = sumGradient; // Reusing neurons for gradient computation
~~~

---

### 🎯 Skipping Low-Error Updates: A Regularization Trick

During training, the network **skips backpropagation for predictions that are already close to the target value**. This is controlled by the `bp` (backpropagation threshold) parameter:

~~~cs
if (Math.Abs(target - prediction) > bp)
{
    // output gradient
    for (int i = 0; i < net[^1]; i++)
        neurons[^(1 + i)] = target - neurons[^(1 + i)];

    Backprop(net, neurons, biasDelta, wt, deltas);
}
~~~

#### Why This is a Good Idea:
1. **Focus on Significant Errors**:
   - Skips updates for predictions close to the target, focusing on larger errors that need correction.
2. **Regularization Effect**:
   - Prevents overfitting to minor fluctuations, improving generalization.
3. **Faster Training**:
   - Reduces unnecessary weight updates, speeding up training without sacrificing accuracy.

---

### 🚀 More Cool Features

This implementation includes several advanced features that make it versatile and robust. Here’s what sets it apart:

1. **Multiple Outputs (Optional)**:
   - The network supports **more than one output neuron**, though this is not typically recommended for regression tasks.
   - When multiple outputs are used, the **mean of the outputs** is often computed as the final prediction. This can sometimes improve stability and accuracy by averaging out noise.

2. **Bias Without Decay**:
   - Unlike weights, **biases do not use decay** during updates. This helps keep the outputs more stable and prevents the model from over-regularizing the bias terms.
   - Biases are updated directly using the learning rate and momentum, ensuring they remain effective in shifting the activation function.

3. **Dynamic Learning Rate and Weight Decay**:
   - The `factor` parameter reduces both the **learning rate** and **weight decay** after each epoch. This gradual reduction helps fine-tune the model as training progresses, leading to better convergence.
   - For example:
     ~~~cs
     lr *= factor; // Learning rate decreases over time
     decay *= factor; // Weight decay decreases over time
     ~~~

4. **Customizable Optimizers**:
   - Choose between **SGD with momentum** and **Adam optimization** for training.
   - SGD with momentum is great for stable convergence, while Adam is ideal for faster training and adaptive learning rates.

---

