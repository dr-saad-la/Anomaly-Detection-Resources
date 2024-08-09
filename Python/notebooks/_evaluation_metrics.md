# Evaluation Metrics 

1. **Area Under the Receiver Operating Characteristic (ROC) Curve (AUC-ROC)**
    - It measures the ability of the model to distinguish between normal and anomalous instances across various threshold settings.
    - **Range:** 0 to 1, with 1 being perfect classification.
    - **Advantage:** Not sensitive to class imbalance.

2. **Precision**:
    - The ratio of correctly identified outliers to the total number of instances predicted 
    - **Formula:** `TP / (TP + FP)`
    - **Use Case:** Useful when the cost of false positives is high.

3. **Recall (Sensitivity)**
    - The ratio of correctly identified outliers to the total number of actual outliers.
    - **Formula:** `TP / (TP + FN)`
    - **Use Case:** Useful when the cost of false negatives is high.

4. **F1-Score**
    - Harmonic mean of precision and recall.
    - **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
    - **Use Case:** Provides a balanced measure between precision and recall.

5. **Average Precision (AP)**
    - Area under the precision-recall curve.
    - **Use Case:** Summarizes the precision-recall curve as a single number.

6. **Matthews Correlation Coefficient (MCC)**
    - It measures the quality of binary classifications.
    - **Range:** -1 to +1, with +1 being perfect prediction.
    - **Advantage:** Handles imbalanced datasets well.

7. **Cohen's Kappa**
    - It Measures agreement between predicted and actual classifications, accounting for agreement by chance.
    - **Range:** -1 to 1, with 1 being perfect agreement.

8. **Specificity**
    - The ratio of correctly identified normal instances to the total number of actual normal instances.
    - **Formula:** `TN / (TN + FP)`

9. **False Positive Rate (FPR)**
    - The ratio of false positives to the total number of actual normal instances.
    - **Formula:** `FP / (FP + TN)`

10. **False Discovery Rate (FDR)**
    - The ratio of false positives to the total number of predicted positives.
    - **Formula:** `FP / (FP + TP)`

11. **Geometric Mean (G-Mean)**

    - The geometric mean of sensitivity and specificity.
    - **Formula:** `sqrt(Sensitivity * Specificity)`
    - **Use Case:** Useful for imbalanced datasets.

12. **Balanced Accuracy**
    - The arithmetic mean of sensitivity and specificity.
    - **Formula:** `(Sensitivity + Specificity) / 2`
    - **Use Case:** Useful for imbalanced datasets.
13. **The Precision at Rank n (P@n)** is an important metric for outlier detection, especially when dealing with **top-n** lists of anomalies. 

- **Precision at Rank n (P@n)**: Measures the precision considering only the top n ranked instances predicted as outliers.
- **Formula**: (Number of true outliers in top n predictions) / n
- **Particularly useful when**:
    a) You have a fixed budget for investigating anomalies
    b) You're more interested in the most anomalous instances
    c) Your dataset has a known number of outliers

- **Characteristics of P@n:**
    - It focuses on the quality of the top predictions, which is often more relevant in practical applications of outlier detection.
    - It's especially useful when the exact number of outliers is unknown, but you can investigate a fixed number of cases.
    - This metric can be calculated at different values of n to understand how precision changes as more predictions are considered.

- **When using P@n:**
    - Choose `n` based on domain knowledge or operational constraints.
    - Consider calculating `P@n` for multiple values of n to get a more comprehensive view of the algorithm's performance.
    - It can be complemented with other metrics like recall or F1-score for a fuller evaluation.

14. **Adjusted Precision at Rank n (Adjusted P@n):**
    - `Adjusted P@n` is an important variation (modification) of the standard `P@n` metric that takes into account the total number of actual outliers in the dataset. It's particularly useful when the number of true outliers is less than `n`.
    - Formula:
      $$\text{Adjusted P@n} = \frac{\text{(Number of true outliers in top n predictions)}} {\text{min(n, total number of true outliers)}}$$
    - **Key characteristics:**
        - It addresses a limitation of standard P@n when n is larger than the actual number of outliers in the dataset.
        - It provides a fairer evaluation when the exact number of outliers is known but is less than n.
        - The adjustment ensures that the metric doesn't unfairly penalize an algorithm for not finding more outliers than actually exist.

    - **When to use Adjusted P@n:**
        - When you know the total number of true outliers in your dataset.
        - In scenarios where n might be larger than the number of actual outliers.
        - When comparing algorithms across datasets with different numbers of outliers.

    - **Advantages:**
        - More robust when dealing with datasets that have few outliers.
        - Provides a more realistic measure of precision in top-n predictions.
        - Allows for fairer comparisons between different outlier detection methods or across different datasets.

> **Note**
The adjusted P@n is an excellent refinement of the P@n metric, providing a more nuanced evaluation in scenarios where the number of true outliers is known and potentially less than n.  It's crucial for a comprehensive and fair evaluation of outlier detection algorithms, especially when dealing with datasets where outliers are rare.


14. **The Maximum F1-Measure**: It is also known as the Optimal F1-Score or F1-Max, is an important evaluation metric for outlier detection algorithms. Here's a detailed explanation:
    - The Maximum F1-Measure is the highest F1-score achievable by an outlier detection algorithm across all possible threshold values.
    - Key characteristics:
        - Threshold-independent: It doesn't rely on a single, predetermined threshold.
        - Balance between Precision and Recall: As an F1-score, it balances the trade-off between precision and recall.
        - Optimal performance indicator: It represents the best possible performance of the algorithm in terms of F1-score.

    - **Calculation process**:

        - Compute precision and recall for various threshold values.
        - Calculate the F1-score for each threshold.
        - Select the maximum F1-score among all thresholds.
    
    - **Formula**:
        $$F1 = 2 * (Precision * Recall) / (Precision + Recall)$$
        $$Maximum F1-Measure = max(F1) across all thresholds$$
    - **Advantages**:

        - Provides a single, interpretable metric that combines precision and recall.
        - Useful for comparing different algorithms without needing to set a specific threshold.
        - Helps in understanding the best possible performance of an algorithm.

    - **Use cases**:

        - Comparing different outlier detection algorithms.
        - Evaluating an algorithm's performance across different datasets.
        - Useful when the optimal threshold is not known in advance.

    - **Considerations**:

        - While it provides the optimal F1-score, the corresponding threshold might not always be practical in real-world applications.
        - It assumes equal importance of precision and recall, which may not always be the case in all scenarios.


> **Note**: The Maximum F1-Measure is particularly valuable in outlier detection because it addresses the challenge of selecting an appropriate threshold, which can be especially difficult when dealing with imbalanced datasets typical in anomaly detection tasks. It offers a comprehensive view of an algorithm's capability to balance between identifying true outliers and avoiding false alarms.

15. **The Adjusted Average Precision (Adjusted AP)**: is an important modification of the standard Average Precision metric, specifically designed for outlier detection scenarios. Here's a detailed explanation:
Definition:
    - **Adjusted AP** is a variation of **Average Precision** that takes into account the total number of actual outliers in the dataset, particularly useful when this number is known and relatively small compared to the dataset size.
    - **Key characteristics:**
    
        - Accounts for limited outliers: It's especially useful when the number of true outliers is known and significantly less than the number of predictions typically considered.
        - Prevents artificial inflation: It addresses the issue of AP being artificially high when there are very few outliers in a large dataset.
        - More realistic evaluation: Provides a more accurate assessment of performance in real-world outlier detection scenarios where outliers are rare.
    
    - **Calculation process:**
    
        - Calculate precision at each correctly identified outlier.
        - Sum these precision values.
        - Divide by the total number of true outliers (instead of the number of predictions as in standard AP).
        
    - **Formula:**
        $$Adjusted AP = (Sum of precision at each correct detection) / (Total number of true outliers)$$
    
    - **Advantages:**
    
        - More appropriate for imbalanced datasets typical in outlier detection.
        - Provides a fairer comparison between different algorithms or datasets.
        - Reflects real-world scenarios where the number of outliers is often known and limited.
    
    - **Use cases**:
    
        - Evaluating outlier detection algorithms in datasets with a known, small number of outliers.
        - Comparing algorithm performance across different datasets with varying outlier ratios.
        - More accurate assessment in fields like fraud detection, where true positives are rare.
    
    - **Considerations**:
    
        - Requires knowledge of the total number of true outliers in the dataset.
        - May give different results compared to standard AP, especially in highly imbalanced scenarios.

> **Note**: The Adjusted AP is particularly valuable in outlier detection because it addresses the limitations of standard AP in scenarios with very few outliers. It provides a more nuanced and realistic evaluation metric, especially important in fields where outliers are rare but critical to identify, such as in fraud detection, system anomaly identification, or rare disease diagnosis.
## Evaluation Considerations

When evaluating outlier detection algorithms, consider the following:
- The nature of your data and the specific problem you're addressing.
- The cost associated with false positives versus false negatives in your context.
- The level of class imbalance in your dataset.
- Whether you need a threshold-independent metric (like AUC-ROC) or a metric at a specific threshold.

It's often beneficial to use multiple metrics to get a comprehensive understanding of the algorithm's performance, as each metric provides a different perspective on the model's effectiveness.


