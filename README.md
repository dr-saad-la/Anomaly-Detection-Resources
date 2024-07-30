# Anomaly Detection Resources

A collection of libraries, frameworks, and tools for anomaly detection. This repository aims to provide a comprehensive resource for developers, researchers, and data scientists working on detecting anomalies in 
various datasets.

It contains resources for anomaly detection in various programming languages, including Python, Julia, Rust, Java and C++. Each subdirectory contains libraries, examples, and tools relevant to the specific language for implementing and understanding anomaly detection algorithms.

**Why this repository?**

As the field of anomaly detection continues to grow, it can be challenging to keep track of the many libraries 
and frameworks available for different tasks and applications. This repository seeks to address that challenge by providing a centralized location for discovering and exploring these resources.

**What's inside?**

This repository contains a curated list of libraries, frameworks, and tools for anomaly detection, including:

* Open-source libraries for implementing various anomaly detection algorithms (e.g., Isolation Forest, Local 
Outlier Factor, etc.)
* Frameworks for building anomaly detection systems (e.g., scikit-learn, TensorFlow, PyTorch, etc.)
* Tools for preprocessing and visualizing data for anomaly detection
* Resources for specific industries or domains (e.g., healthcare, finance, IoT, etc.)

## Contributing: **How can I contribute?**

This repository is open to contributions! If you have a favorite library, framework, or tool for anomaly detection, examples, or resources related to anomaly detection in Python, Julia, Rust, Java or C++, feel free to submit a pull request or open an issue, please share it with the community by creating a pull request. We welcome additions, updates, and suggestions on how to make this repository more valuable.


### How to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch and create a pull request.


## Usage: **How can I use this repository?**

You can use this repository as a starting point for your own projects, or as a reference guide for learning 
about different libraries and frameworks for anomaly detection. Whether you're a beginner looking to get 
started with anomaly detection or an experienced practitioner seeking new tools and techniques, this 
repository aims to provide something of value.


## Directory Structure

- **Python/**: Libraries and tools for anomaly detection using Python.
- **Julia/**: Resources for implementing anomaly detection in Julia.
- **Rust/**: Libraries and examples for anomaly detection in Rust.
- **Java/**: Libraries and examples for anomaly dectection in Java.
- **C++/**: Tools and libraries for anomaly detection using C++.

## Contents

### Python Libraries for Anomaly Detection

1. [PyOD](https://github.com/yzhao062/pyod)
PyOD is a comprehensive and scalable Python library for detecting outlying objects in multivariate data. It includes over 50 detection algorithms, such as the Isolation Forest, k-Nearest Neighbors (kNN), and AutoEncoder. It supports both classical and deep learning models, making it suitable for a wide range of anomaly detection tasks. [PyOD docs](https://pyod.readthedocs.io/en/latest/)

2. [PyGOD (Python Graph Outlier Detection)](https://github.com/pygod-team/pygod)
Graph based outlier detection library. It includes algorithms designed to detect anomalies in graph data, such as social networks or security systems. PyGOD leverages PyTorch and PyTorch Geometric to provide efficient implementations of various graph-based anomaly detection methods. [PyGOD docs](https://docs.pygod.org/en/latest/)

3. [Time-series Outlier Detection (TODS)](https://tods-doc.github.io/)

4. [Automating Outlier Detection via Meta-Learning (MetaOD)](https://github.com/yzhao062/MetaOD?tab=readme-ov-file)

5. [ADTK (Anomaly Detection Toolkit)](https://github.com/arundo/adtk)
Unsupervised/rule-based time series anomaly detection. It provides tools for detecting anomalies based on seasonal patterns, trend deviations, and other customizable rules. ADTK is highly flexible and suitable for various time series anomaly detection applications.

6. [Anomalib](https://github.com/openvinotoolkit/anomalib)
A Deep learning library that collects state-of-the-art anomaly detection algorithms for benchmarking on public and private datasets. It provides ready-to-use implementations of various anomaly detection algorithms and tools for developing and deploying custom models. Anomalib is especially useful for visual anomaly detection.

7. [scikit-learn](https://scikit-learn.org/stable/modules/outlier_detection.html)
It includes several tools for anomaly detection, such as the One-Class SVM, Isolation Forest, and Elliptic Envelope. These methods can be applied to identify outliers in datasets and are integrated with the rest of the scikit-learn ecosystem for easy use in machine learning pipelines.


### Python Libraries for Outlier (Anomaly) Detection

- **Libraries**: Lists and descriptions of Python libraries used for anomaly detection:
	1. [Python for Outlier Detection (PyOD)](https://pyod.readthedocs.io/en/latest/)
	2. [Python for Graph Outlier Detection(PyGOD)]()
	

### Python Examples:

This section contains:

1. **Example scripts**: demonstrating the use of various libraries for anomaly detection tasks.
2. **Notebooks**: Jupyter notebooks for interactive anomaly detection experiments.



### Julia Libraries for Anomaly Detection

This section contains the available in Julia libraries for anomaly detection: 

1. [AnomalyDetection.jl](https://github.com/smidl/AnomalyDetection.jl)
It provides tools for detecting anomalies in time-series data. It includes both univariate and multivariate anomaly detection methods.

2. [OutlierDetection.jl](https://github.com/OutlierDetectionJL/OutlierDetection.jl)
This offers a collection of outlier detection methods for both univariate and multivariate data. It includes techniques such as the Isolation Forest, One-Class SVM, and Local Outlier Factor (LOF).

3. [Boltzmann.jl](https://github.com/dfdx/Boltzmann.jl)
While primarily a library for energy-based models, Boltzmann.jl can also be used for anomaly detection tasks by modeling the probability distribution of normal data and identifying deviations from it.

4. [HDBSCAN.jl](https://github.com/baggepinnen/HDBSCAN.jl)
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is primarily a clustering algorithm, but it can also be used for anomaly detection by identifying points that do not belong to any cluster (noise points). [Python Documentation is here](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html)

5. [ClusterValidityIndices.jl](https://github.com/AP6YC/ClusterValidityIndices.jl)
Although this package is focused on cluster validity indices, it can be useful in anomaly detection to validate the presence of anomalous clusters within a dataset [oai_citation:1,GitHub - AP6YC/ClusterValidityIndices.jl: A Julia package for Cluster Validity Indices (CVIs).](https://github.com/AP6YC/ClusterValidityIndices.jl) [oai_citation:2,GitHub - AP6YC/ICVI-Examples: Example usage of the Incremental Cluster Validity Indices (ICVI) implemented in the AdaptiveResonance.jl julia package.](https://github.com/AP6YC/ICVI-Examples).

6. [GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl)
A library for modeling data using Gaussian Mixture Models (GMMs). Anomalies can be detected by identifying data points with low likelihood under the fitted GMM.

7. [MultivariateAnomalies.jl](https://github.com/milanflach/MultivariateAnomalies.jl/tree/master)
This package is designed for detecting anomalies in multivariate data using various statistical methods. It provides a robust framework for anomaly detection in high-dimensional datasets.

### Julia Examples for Anomaly Detection

This section contains example scripts showcasing anomaly detection implementations in Julia.


### Resources to Learning Julia for Anomaly Detection

1. [Anomaly Detection Algorithms](https://milanflach.github.io/MultivariateAnomalies.jl/stable/man/DetectionAlgorithms/)

### Rust 

Rust libraries and crates for anomaly detection: 


1. [rustlearn](https://github.com/maciejkula/rustlearn)
Rustlearn is a machine learning library in Rust that includes various algorithms, including tools for anomaly detection. It is focused on performance and ease of use.

2. [linfa](https://github.com/rust-ml/linfa)
Linfa is a comprehensive machine learning library for Rust that provides tools for a wide range of tasks, including anomaly detection. It is inspired by Python's scikit-learn.

3. [smartcore](https://github.com/smartcorelib/smartcore)
SmartCore is a Rust library for machine learning that includes various algorithms for anomaly detection, such as One-Class SVM and Isolation Forest.

4. [ndarray-stats](https://github.com/rust-ndarray/ndarray-stats)
A Rust library that extends `ndarray` with statistical methods, including tools for detecting anomalies in data. It is useful for data analysis and preprocessing.

### Rust Examples for Anomaly Detection

1. **Examples**: Sample projects and code snippets demonstrating the use of Rust for anomaly detection.
2. **Guides**: Step-by-step guides on setting up and using Rust libraries for anomaly detection.



### C++ Libraries for Anomaly Detection

1. [Anomalib](https://github.com/openvinotoolkit/anomalib)
Anomalib is a deep learning library designed for anomaly detection, offering a collection of state-of-the-art algorithms. It focuses on visual anomaly detection, providing tools for developing, training, and deploying anomaly detection models. The library includes a modular API and CLI for ease of use, supporting various models and benchmarks.

2. [AnomalyDetection.cpp](https://github.com/ankane/AnomalyDetection.cpp)
This library is a C++ port of the AnomalyDetection R package, designed for time series anomaly detection. It provides methods for detecting anomalies in time-series data, with configurable parameters like statistical significance level and maximum anomalies percentage. It uses seasonal-trend decomposition and quantile functions for its calculations.

3. [Anomaly Detection in C++](https://github.com/mtrazzi/anomaly-detection-in-cpp)
This implementation is based on an anomaly detection project from Andrew Ng's machine learning course on Coursera. It uses a multivariate normal distribution to estimate the probability distribution of the data and computes the best threshold epsilon for classifying data points as anomalies.

4. [OutlierTree](https://github.com/david-cortes/outliertree)
OutlierTree is an explainable outlier/anomaly detection library that uses decision tree conditioning. It is similar to the GritBot software and supports various data types, handling missing values effectively. The library is written in C++ and can be interfaced with Python and R. It provides human-readable justifications for flagged outliers, making it ideal for exploratory data analysis.



### C++ Examples 

- **Examples**: Example code and projects for performing anomaly detection using C++.


### Resources:




### Java Libraries for Anomaly Detection 

# Java Libraries for Anomaly Detection

## 1. [EGADS (Extensible Generic Anomaly Detection System)](https://github.com/yahoo/egads)
EGADS is an open-source Java package developed by Yahoo for automatically detecting anomalies in large-scale time-series data. It supports a variety of anomaly detection techniques and can be integrated into existing monitoring infrastructures. The library separates the time-series modeling and anomaly detection components, allowing for flexible customization of models.

## 2. [Tribuo](https://tribuo.org/)
Tribuo is a comprehensive machine learning library for Java that includes anomaly detection capabilities. It provides infrastructure for anomaly detection tasks and supports one-class SVM (LibSVM) and one-class linear SVM (LibLinear). Tribuo also offers a rich set of features for classification, regression, and clustering, making it a versatile tool for various machine learning applications.

## 3. [Java Anomaly Detection Library](https://github.com/ExpediaDotCom/adaptive-alerting)
The Adaptive Alerting library by Expedia provides tools for anomaly detection and alerting in time-series data. It includes several built-in anomaly detection models, such as the Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD) and Holt-Winters models, and supports the addition of custom models.



### Java Examples for Anomaly Detection 


### Java Additional Resources 



## Anomaly Detection Resources 

1. [Github Repo for Anomaly Detection Resources](https://github.com/yzhao062/anomaly-detection-resources)


## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

