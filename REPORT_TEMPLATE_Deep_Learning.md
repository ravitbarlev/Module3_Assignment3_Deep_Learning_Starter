# REPORT — Module 3 · Assignment 3 · Deep Learning Foundations

**Name:**Ravit Bar-Lev_  **ID:** 029290400  **Date:** 01/08/2026
**Chosen option:** ___ B · Fashion-MNIST CNN 

> Keep this report in English. A neural net that loses to a simpler model is a finding,
> not a failure. Report it honestly.

---

## 1. Framing
Task and primary metric:
The task was to practice SmallCNN and class MLP on FashionMNIST that incudes 60K clothes picture 
divided to 10 classes. The training part focuses on accuracy by comparing the results to 
real tags. This analysis presents the learning curve of pictures identification model.

The architecture used is: layers & params
ץ
Layers: SmallCNN - containing 2 Convolutional layers, 2 Max Pooling layers and 2 Fully Connected layers.

Params:206,922 total trainable parameters

Baseline you are comparing against:
The baseline model I am comparing against is the logistic regression.
---

## 2. Results
| Model | Test metric (accuracy) | Params | Train time | Notes |
|---|---|---|---|---|
| simpler baseline |83.50% |7850 |a few seconds  |simple linear model  |
| neural net |91.77% |206922 | 12 minutes| SmallCNN demonstrated overfitting|

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Did DL win?** Did your neural net beat the simpler / cheaper baseline? By how much, and at what cost?
   Yes, Deep Learning won decisively by improving accuracy by **~8.7%**.
   The costs include a much higher parameter footprint (206k params vs. hundreds), longer training time
   (~70 seconds per epoch), higher compute requirements (GPU/CUDA preferred), and a complete loss of direct feature
   interpretability (black-box model)
   
2. **Logits / loss.** Which loss did you use and why? What breaks if you apply softmax / sigmoid before it?
   I used CrossEntropyLoss because it internally applies LogSoftMax.
   Following the project instructions, no Softmax layer was applied at the end of the network.
   
3. **Overfitting.** Read your learning curves. Where do train and val diverge, and what did you do about it?
   Overfitting diagnostic shows divergence at **Epoch 5**. Up to epoch 5, both losses decrease (`Train Loss: 0.1810`,
   `Val Loss: 0.2417`). From epoch 6 onwards, `Train Loss` continuous to sink down to `0.1027` while `Val Loss`
   degrades and climbs back up to `0.2518`, signaling over-parameterization
   
4. **Learning rate.** Show what happens with a too-large and a too-small LR. Why is it the most important know?
    Tested across 3 orders of magnitude with identical computation times (~350 seconds per 5 epochs):
  - `LR = 0.1` (Too Large): Failed completely. Loss stuck at `2.30` and accuracy at `10.4%`
    (equivalent to random guessing), due to exploding gradients skipping the minimum.
  - `LR = 0.001` (Optimal): Smooth, rapid convergence achieving `91.05%` validation accuracy by epoch 5.
  - `LR = 1e-05` (Too Small): Extremely slow convergence, reaching only `79.27%` accuracy by epoch 5,
      requiring vastly more epochs to converge. 
   
5. **Regularization.** Which regularizer did you use, and did it actually help (with numbers)?
    Adam` optimizer with an optimal learning rate of `1e-3` (`0.001`). Dropout / weight penalty act
   as regularizers to control variance
   
6. **Cost / benefit.** Count the cost of the NN (params, training time, interpretability) vs the simple model. Is it justified?
    It is all a matter of trade-off. The initial goal was to 
    get an accurate model as much as possible, however, the price is high.
   
7. **When DL.** For your task, would you deploy DL or the simpler model? Defend it.
    Deploy the DL model (SmallCNN).** For an e-commerce or fashion classification application, a ~9%
     boost in accuracy (reducing errors by more than half) directly translates to a significantly better
     user experience and fewer misclassified items. Since the model is lightweight (under 210k parameters), the inference
     latency is negligible, justifying the computational overhead during training
   
8. **Monday morning.** What would you monitor in production, and what would trigger a retrain?
    Inference latency (ms per batch), prediction distribution drift (e.g., if the ratio of "T-shirt"
    predictions suddenly spikes), data drift in input image distributions (brightness, contrast, resolution),
   and user-reported misclassification rates
---

## 4. DL Model Card
Paste the completed DL Model Card from the notebook here.

## 1. Overview
- **Option / task / data:** Multi-class classification (10 classes) of fashion items using the FashionMNIST dataset (60,000 grayscale images of size 28x28).
- **Architecture (layers, params):** 
  - **Layers:** `SmallCNN` containing 2 Convolutional layers (`Conv2d` with 16 and 32 filters, 3x3 kernel, padding=1), 2 Max Pooling layers (`MaxPool2d`, 2x2), and 2 Fully Connected layers (`Linear` with 128 hidden units and 10 output units).
  - **Params:** 206,922 total trainable parameters.

## 2. Setup
- **Loss and why (logits handling):** `CrossEntropyLoss`. It internally applies `LogSoftmax` and `NLLLoss` in a single, numerically stable step. Following project instructions, no Softmax layer was applied at the end of the network, leaving the output as raw logits.
- **Optimizer, learning rate, regularizer:** `Adam` optimizer with an optimal learning rate of `1e-3` (`0.001`). Dropout / weight penalty act as regularizers to control variance.

## 3. Performance
- **Simpler-model baseline:** A standard shallow/linear baseline (e.g., Logistic Regression / Decision Tree) yields around ~83.0% validation accuracy on this dataset.
- **Neural-net test metric:** Achieved a peak validation accuracy of **91.77%** (and 91.05% within the first 5 epochs).
- **Did DL win? By how much, at what cost?:** Yes, Deep Learning won decisively by improving accuracy by **~8.7%**. The costs include a much higher parameter footprint (206k params vs. hundreds), longer training time (~70 seconds per epoch), higher compute requirements (GPU/CUDA preferred), and a complete loss of direct feature interpretability (black-box model).

## 4. Diagnostics
- **Learning curves: where do train and val diverge?:** Overfitting diagnostic shows divergence at **Epoch 5**. Up to epoch 5, both losses decrease (`Train Loss: 0.1810`, `Val Loss: 0.2417`). From epoch 6 onwards, `Train Loss` continuous to sink down to `0.1027` while `Val Loss` degrades and climbs back up to `0.2518`, signaling over-parameterization.
- **Learning-rate sensitivity:** Tested across 3 orders of magnitude with identical computation times (~350 seconds per 5 epochs):
  - `LR = 0.1` (Too Large): Failed completely. Loss stuck at `2.30` and accuracy at `10.4%` (equivalent to random guessing), due to exploding gradients skipping the minimum.
  - `LR = 0.001` (Optimal): Smooth, rapid convergence achieving `91.05%` validation accuracy by epoch 5.
  - `LR = 1e-05` (Too Small): Extremely slow convergence, reaching only `79.27%` accuracy by epoch 5, requiring vastly more epochs to converge.

## 5. Decision
- **Would you deploy DL or the simpler model here? Defend it:** **Deploy the DL model (SmallCNN).** For an e-commerce or fashion classification application, a ~9% boost in accuracy (reducing errors by more than half) directly translates to a significantly better user experience and fewer misclassified items. Since the model is lightweight (under 210k parameters), the inference latency is negligible, justifying the computational overhead during training.
- **Production: what to monitor, what triggers a retrain:**
  - **What to monitor:** Inference latency (ms per batch), prediction distribution drift (e.g., if the ratio of "T-shirt" predictions suddenly spikes), data drift in input image distributions (brightness, contrast, resolution), and user-reported misclassification rates.
  - **What triggers a retrain:** A sustained drop in classification accuracy, introducing a new seasonal clothing line (data drift), or when a predefined threshold of feedback errors is crossed.

```
(DL Model Card)
```

---

## 5. Reflection
What surprised you? When, in your mid-term project, would reaching for DL be the right call?
The fact that it had reached a very high level of accuracy with comparing the baseline model.
The "cost" of this accuracy is high.
In the mid project, we would stick to the baseline model cause our sample is small
