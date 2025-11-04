# **Q1. Explain the architecture and working of an Autoencoder. Derive the mathematical formulation for the encoding and decoding processes, and illustrate how the reconstruction loss is minimized.**

---

## **Answer:**

### **1️⃣ Definition:**

An **Autoencoder** is a type of **neural network** used for **unsupervised learning** that aims to **learn a compressed (latent) representation** of the input data and then **reconstruct it** as accurately as possible.

It consists of two main parts:

- **Encoder:** Compresses the input into a lower-dimensional representation (latent space)
- **Decoder:** Reconstructs the input from this compressed representation

---

### **2️⃣ Architecture:**

```
Input → Encoder → Bottleneck → Decoder → Output (Reconstruction)
```

- The **bottleneck layer** is the **compressed latent space** that forces the model to learn only the most **relevant features** of the input data

---

### **3️⃣ Working Principle:**

#### **Encoder:**
Maps input vector \( x \) to a lower-dimensional latent vector \( h \):
\[
h = f(W_e x + b_e)
\]
where  
\( W_e \) = encoder weight matrix,  
\( b_e \) = bias,  
\( f \) = activation function (e.g., ReLU, sigmoid)

#### **Decoder:**
Reconstructs the original input from \( h \):
\[
\hat{x} = g(W_d h + b_d)
\]
where  
\( W_d \) = decoder weight matrix,  
\( b_d \) = bias,  
\( g \) = output activation function

---

### **4️⃣ Loss Function:**

The goal is to make \( \hat{x} \) as close as possible to \( x \).  
The **Reconstruction Loss** is minimized:
\[
L(x, \hat{x}) = \|x - \hat{x}\|^2 = \sum_i (x_i - \hat{x}_i)^2
\]

👉 For **binary inputs**, the **Binary Cross-Entropy (BCE)** loss can be used:
\[
L(x, \hat{x}) = -\sum_i [x_i \log(\hat{x}_i) + (1 - x_i)\log(1 - \hat{x}_i)]
\]

---

### **5️⃣ Optimization:**

Weights \( W_e, W_d \) are updated using **backpropagation** and **gradient descent** to minimize \( L(x, \hat{x}) \)

---

### **6️⃣ Key Idea:**

The **bottleneck layer** forces the network to **learn a compact, meaningful representation** — effectively performing **dimensionality reduction** similar to PCA, but **non-linear** and **data-driven**

---

### **Applications:**

- Image denoising
- Dimensionality reduction
- Feature learning
- Machine translation (Encoder–Decoder sequence models)

---

---

# **Q2. What are Regularized Autoencoders? Compare Sparse Autoencoder, Denoising Autoencoder, and Variational Autoencoder (VAE) in terms of their architecture, regularization techniques, and objectives.**

---

## **Answer:**

### **1️⃣ Regularized Autoencoders:**

These are **modified Autoencoders** that include **constraints or noise** during training to prevent the network from simply copying the input and to make the learned representations more **robust and meaningful**

---

## **Types of Regularized Autoencoders:**

| **Type** | **Key Idea / Architecture** | **Regularization Technique** | **Objective / Effect** |
|----------|-----------------------------|------------------------------|------------------------|
| **Sparse Autoencoder (SAE)** | Hidden units produce sparse activations (only a few neurons active at a time) | Adds a **sparsity penalty** using **Kullback-Leibler (KL) divergence** between average activation \( \hat{\rho} \) and target sparsity \( \rho \): <br> \[ \Omega = \sum_j KL(\rho \parallel \hat{\rho_j}) = \sum_j [\rho \log\frac{\rho}{\hat{\rho_j}} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho_j}}] \] | Forces neurons to learn distinct, meaningful features |
| **Denoising Autoencoder (DAE)** | Input is **corrupted with noise**, but the model must reconstruct the original clean input | Randomly corrupt input \( \tilde{x} = x + \text{noise} \). Train model to minimize: <br> \[ L = \|x - \hat{x}\|^2 \] | Learns **robust features** that are resistant to noise or missing data |
| **Variational Autoencoder (VAE)** | Encoder outputs a **distribution** (mean & variance) instead of a fixed vector | Adds **KL divergence** between learned latent distribution \( q(z\|x) \) and prior \( p(z) = N(0,1) \): <br> \[ L = E_{q(z\|x)}[\log p(x\|z)] - KL(q(z\|x) \| p(z)) \] | Learns **continuous, generative latent space** to produce new data samples. Uses **reparameterization trick** to allow backpropagation through random sampling |

---

### **2️⃣ Reparameterization Trick (for VAE):**

To make sampling differentiable:
\[
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim N(0, I)
\]
This allows gradients to flow through \( \mu \) and \( \sigma \) during training

---

### **3️⃣ Summary:**

| **Autoencoder Type** | **Goal** | **Special Property** |
|---------------------|----------|----------------------|
| Sparse AE | Learn compact, interpretable features | Enforces sparsity using KL divergence |
| Denoising AE | Learn noise-robust representation | Reconstructs clean input from noisy data |
| Variational AE | Learn generative latent space | Produces new samples, uses reparameterization |

---

### **4️⃣ Key Takeaways:**

- Regularization prevents overfitting and helps extract **useful representations**
- **VAE** is the foundation for **modern generative models** like **GANs** and **Transformers**
- These autoencoders are used in **feature extraction**, **image restoration**, **data generation**, and **representation learning**
