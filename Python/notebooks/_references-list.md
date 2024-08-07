# Anomaly Detection References

This document provides a list of references for outlier detection algorithms. Note that this list will expand as we move forward through our series  of lectures. 

## References

### Academic Articles

1. **Kriegel, H.-P., Kröger, P., Schubert, E., & Zimek, A. (2008).** *Angle-Based Outlier Detection in High-dimensional Data*. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 444-452). ACM.  
   [DOI:10.1145/1401890.1401946](https://dl.acm.org/doi/10.1145/1401890.1401946)  
   - This is the original paper introducing the ABOD algorithm, detailing the method's approach to identifying outliers in high-dimensional spaces using angular relationships.

2. **Zimek, A., Schubert, E., & Kriegel, H.-P. (2012).** *A survey on unsupervised outlier detection in high-dimensional numerical data*. Statistical Analysis and Data Mining: The ASA Data Science Journal, 5(5), 363-387.  
   [DOI:10.1002/sam.11161](https://onlinelibrary.wiley.com/doi/abs/10.1002/sam.11161)  
   - This survey paper explores various unsupervised outlier detection methods, with insights into high-dimensional challenges, including discussions on ABOD.

3. **Schubert, E., Zimek, A., & Kriegel, H.-P. (2012).** *Local Outlier Detection Reconsidered: A Generalized View on Locality with Applications to Spatial, Video, and Network Outlier Detection*. Data Mining and Knowledge Discovery, 28(1), 190-237.  
   [DOI:10.1007/s10618-012-0300-z](https://link.springer.com/article/10.1007/s10618-012-0300-z)  
   - This paper provides a generalized view on local outlier detection methods, offering insights into various applications and theoretical foundations.

4. **Zhang, K., Hutter, M., & Jin, H. (2009).** *A new local distance-based outlier detection approach for scattered real-world data*. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 813-822). Springer.  
   [DOI:10.1007/978-3-642-01307-2_87](https://link.springer.com/chapter/10.1007/978-3-642-01307-2_87)  
   - This paper discusses local distance-based outlier detection, offering an alternative perspective that complements the angle-based approach of ABOD.

5. **Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000).** *LOF: Identifying Density-Based Local Outliers*. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data (pp. 93-104). ACM.  
   [DOI:10.1145/342009.335388](https://dl.acm.org/doi/10.1145/342009.335388)  
   - Although focused on the Local Outlier Factor (LOF) method, this paper provides valuable insights into density-based outlier detection, which is often contrasted with angle-based methods.

6. **Hodge, V. J., & Austin, J. (2004).** *A survey of outlier detection methodologies*. Artificial Intelligence Review, 22(2), 85-126.  
   [DOI:10.1023/B:AIRE.0000045502.10941.a9](https://link.springer.com/article/10.1023/B:AIRE.0000045502.10941.a9)  
   - A comprehensive survey of outlier detection methodologies, this paper provides context for understanding where ABOD fits within the larger landscape of outlier detection techniques.

### Books

1. **Aggarwal, C. C. (2013).** *Outlier Analysis*. Springer.  
   [ISBN:978-1-4614-6395-6](https://link.springer.com/book/10.1007/978-1-4614-6396-3)  
   - This book provides comprehensive coverage of various outlier detection methods, including a section on angle-based techniques. It is a great resource for understanding the broader context of outlier analysis.

2. **Tan, P.-N., Steinbach, M., & Kumar, V. (2005).** *Introduction to Data Mining*. Pearson.  
   [ISBN:978-0321321367](https://www.pearson.com/us/higher-education/program/Tan-Introduction-to-Data-Mining-1st-Edition/PGM207437.html)  
   - This textbook provides a foundational understanding of data mining techniques, including sections on outlier detection.

### Online Documentation and Resources

1. **PyOD: A Python Toolbox for Scalable Outlier Detection (2019).**  
   [GitHub Repository](https://github.com/yzhao062/pyod)  
   - PyOD is a comprehensive Python library for outlier detection. The library implements the ABOD algorithm and provides documentation and examples for practical applications.

2. **PyOD Documentation**  
   [PyOD Documentation](https://pyod.readthedocs.io/en/latest/)  
   - The PyOD documentation provides practical examples and usage details for applying the ABOD algorithm and other outlier detection methods in Python.

3. **Wikipedia Article on Outlier Detection**  
   [Outlier Detection - Wikipedia](https://en.wikipedia.org/wiki/Anomaly_detection)  
   - This Wikipedia article gives an overview of anomaly detection techniques, including angle-based methods like ABOD.


### Additional Resources

- **Medium: Anomaly Detection Techniques: A Comprehensive Guide with Supervised and Unsupervised Learning**  
  [Medium](https://medium.com/@venujkvenk/anomaly-detection-techniques-a-comprehensive-guide-with-supervised-and-unsupervised-learning-67671cdc9680)  
  - This article provides a comprehensive guide on anomaly detection techniques, covering both supervised and unsupervised learning methods with practical examples.

- **Data Head Hunters: How to Use Python for Anomaly Detection in Data: Detailed Steps**  
  [Data Head Hunters](https://dataheadhunters.com/academy/how-to-use-python-for-anomaly-detection-in-data-detailed-steps/)  
  - This article provides a detailed guide on using Python for anomaly detection, covering the necessary steps and offering practical examples to implement various techniques.

- **Analytics Vidhya: Outlier Detection Python - PyOD**  
  [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/)  
  - A blog post exploring outlier detection techniques using PyOD, including the ABOD algorithm, with Python code examples and insights into practical applications.

- **Coursera: Anomaly Detection in Machine Learning**  
  [Coursera](https://www.coursera.org/articles/anomaly-detection-machine-learning)  
  - This article on Coursera provides an overview of anomaly detection in machine learning, discussing various techniques, their applications, and the importance of anomaly detection in different domains.
 
- **Analytics Vidhya: Learning Different Techniques of Anomaly Detection**  
  [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2023/01/learning-different-techniques-of-anomaly-detection/)  
  - This article explores various anomaly detection techniques, offering insights into different algorithms and practical tips for implementing these methods in real-world scenarios.
 
- **KDNuggets: Beginner’s Guide to Anomaly Detection Techniques in Data Science**  
  [KDNuggets](https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html)  
  - This beginner's guide covers anomaly detection techniques in data science, explaining the fundamentals and offering practical insights into implementing these techniques effectively.

- **KDNuggets: An Introduction to Anomaly Detection**  
  [KDNuggets](https://www.kdnuggets.com/2017/04/datascience-introduction-anomaly-detection.html)  
  - This introduction to anomaly detection discusses the concept, significance, and various methods for identifying anomalies, providing a solid foundation for those new to the field.