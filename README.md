# URL-Detection-Application---Machine-Learning-FNN-
URL Detection Application using Machine Learning (FNN)

## **Introduction**

The **URL Detection App** is an Android application developed using **Kotlin with Jetpack Compose** and **TensorFlow Lite**, designed to perform **on-device detection of malicious URLs and QR codes**.  
Its primary goal is to **protect users from phishing and malicious links** without requiring them to open or interact with suspicious URLs or QR codes.

The application offers two main input modes:
1. **Manual URL Entry** – Users can type or paste a URL for analysis.  
2. **QR Code Scanning** – Users can scan QR codes containing embedded links.  

---

## Application UI/UX
<img width="334" height="687" alt="image" src="https://github.com/user-attachments/assets/9cfac368-5603-4a5d-8618-15e65b326156" />
<img width="334" height="687" alt="reuslt_page" src="https://github.com/user-attachments/assets/5158ca75-8a71-4a7b-aaad-be1cffb0030a" />


---

## **Model Overview**

The detection model is built using a **Feed-Forward Neural Network (FNN)** — a type of **multi-layer perceptron** implemented via the **Keras deep learning API**, and later converted into a **TensorFlow Lite (.tflite)** format for efficient deployment on Android devices.

Before training, the raw URL dataset undergoes **feature engineering** and **text vectorization** to extract meaningful numerical representations suitable for neural network learning.

---

### **Feature Engineering**

Referring to the dataset methodology described in DOI: [10.17632/hx4m73v2sf.2](https://doi.org/10.17632/hx4m73v2sf.2), several static URL-based features are derived to capture key URL characteristics that can differentiate benign from malicious links.  

| **Feature** | **Description** |
|--------------|----------------|
| `url_length` | Total number of characters in the URL |
| `has_ip_address` | Binary flag (1/0): whether the URL contains an IP address |
| `dot_count` | Number of “.” characters in the URL |
| `https_flag` | Binary flag (1/0): whether the URL uses HTTPS |
| `url_entropy` | Shannon entropy of the URL (higher values = more randomness) |
| `token_count` | Number of tokens/words in the URL |
| `subdomain_count` | Number of subdomains in the URL |
| `query_param_count` | Number of query parameters after “?” |
| `tld_length` | Length of the top-level domain (e.g., “.com” = 3) |
| `path_length` | Length of the path segment after the domain |
| `has_hyphen_in_domain` | Binary flag: whether the domain contains a hyphen (-) |
| `number_of_digits` | Count of numeric characters in the URL |
| `tld_popularity` | Binary flag: whether the TLD is a popular one |
| `suspicious_file_extension` | Binary flag: whether the URL ends with suspicious extensions (.exe, .zip, etc.) |
| `domain_name_length` | Length of the domain name |
| `percentage_numeric_chars` | Percentage of numeric characters in the URL |

These features collectively represent the **static characteristics of URLs**, allowing the model to infer suspicious structural patterns without needing live network requests or dynamic execution.

---

### **Text Vectorization Using TF-IDF**

In addition to the handcrafted statistical features, each URL string is also converted into a **numerical vector** using the **Term Frequency–Inverse Document Frequency (TF-IDF)** technique.  

- The **TF-IDF vectorizer** operates on **character-level n-grams (3–5 characters)**.  
- This captures subtle textual patterns and obfuscation tactics used in phishing URLs (e.g., “paypa1” instead of “paypal”).  
- The resulting vectors (up to 5000 features) serve as the model’s **input layer**.

This hybrid of **statistical + TF-IDF features** enhances the model’s ability to generalize across unseen URLs.

---

### **Neural Network Architecture**

The final model architecture is a lightweight **Feed-Forward Neural Network (FNN)** optimized for mobile inference.

| **Layer** | **Description** |
|------------|----------------|
| Input Layer | Accepts TF-IDF feature vectors |
| Dense (64 units, ReLU) | Learns high-level URL representations |
| Dense (32 units, ReLU) | Extracts nonlinear patterns from encoded features |
| Dense (1 unit, Sigmoid) | Outputs probability of the URL being malicious |

The model is trained using **binary cross-entropy loss** and the **Adam optimizer**, with **early stopping** to prevent overfitting.  
After training, the model is converted to **TensorFlow Lite (.tflite)** format for deployment in the Android app.

---

## **Dataset**

The model training and evaluation are based on the **Grambeddings Dataset**:  
🔗 [https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/](https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/)  

This dataset contains a balanced mix of **benign and phishing URLs**, allowing robust model generalization and accurate malicious URL detection.

---

## **Results**

### **5-Fold Cross-Validation Results**

| **Fold** | **Best Epoch** | **Validation Accuracy** | **Validation F1-Score** | **Training Notes** |
|-----------|----------------|--------------------------|--------------------------|--------------------|
| Fold 1 | 3 | 0.9615 | 0.9620 | Early stopped at epoch 6 (best weights from epoch 3) |
| Fold 2 | 3 | 0.9601 | 0.9603 | Early stopped at epoch 6 (best weights from epoch 3) |
| Fold 3 | 2 | 0.9595 | 0.9600 | Early stopped at epoch 5 (best weights from epoch 2) |
| Fold 4 | 3 | 0.9613 | 0.9616 | Early stopped at epoch 6 (best weights from epoch 3) |
| Fold 5 | 3 | 0.9615 | 0.9620 | Early stopped at epoch 6 (best weights from epoch 3) |

### **Cross-Validation Summary**

| **Metric** | **Average** | **Standard Deviation** |
|-------------|-------------|------------------------|
| Accuracy | **0.9607** | ±0.0008 |
| F1-Score | **0.9612** | ±0.0009 |


### **Final Model (Trained on Full Dataset)**

| **Epoch** | **Validation Accuracy** | **Validation Loss** | **Observation** |
|------------|--------------------------|---------------------|-----------------|
| 1 | 0.9554 | 0.1149 | Initial convergence |
| 2 | 0.9606 | 0.1040 | Significant improvement |
| 3 | 0.9622 | 0.1015 | Peak performance (best weights) |
| 4 | 0.9632 | 0.1054 | Slight overfitting begins |
| 5 | 0.9623 | 0.1136 | Stable but no improvement |
| 6 | 0.9624 | 0.1204 | Early stopping triggered (best epoch: 3) |

**Final Selected Model Performance (Best Epoch = 3):**
- **Validation Accuracy:** 0.9622  
- **Validation F1-Score:** ≈ 0.962  
- **Loss:** 0.1015  
- **Training Time:** ~90–100 seconds per epoch  

---
