# **Glacier Movement Prediction with Attention-based Recurrent Neural Networks and Satellite Data**  
![Glacier Movement Animation](animationParvatialigned.gif)  

This repository provides tools for extracting satellite images from Microsoft's Planetary Computer database, preprocessing them with enhanced correlation coefficient maximization, and training recurrent neural networks (RNNs) for glacier movement prediction. The methodology includes a novel approach to dividing images into sequential patches for deep learning applications, as detailed in [paper_reference].  

## **Setup and Installation**  
To run the code, it is recommended to use a virtual environment and install the required dependencies using:  
```bash
pip install -r requirements.txt
```

---

## **1. Satellite Image Extraction**  
The code primarily uses **Landsat-8** images but can be adapted for other satellite datasets in the Planetary Computer.  

### **Key Components:**  
- **`config.py`**: The main configuration file where all hyperparameters for image extraction and alignment are specified.  
- **`getCoordinatesFromMap.ipynb`**: A Jupyter notebook that provides an interactive map (using `ipyleaflet`) to extract bounding boxes for specific regions of interest based on latitude and longitude.  
- **`dataAPI.py`**: Handles the extraction of satellite images, storing them as pickle files for each year in structured folders under `/datasets`.  
- **`alignment.py`**: Aligns images and extracts relevant spectral bands. If **NDSI maps** (useful for glacier, ice, and snow analysis) should be generated, enable the `extractNDSI` flag in `config.py`.  

---

## **2. Patching for Deep Learning Applications**  
The **`createPatches.py`** script extracts tensors containing sequential patches from the same coordinates across different time steps. This methodology is explained in detail in [[paper_link](https://www.researchgate.net/profile/Jonas-Mueller-17/publication/376807637_Glacier_Movement_Prediction_with_Attention-based_Recurrent_Neural_Networks_and_Satellite_Data/links/658983872468df72d3d82576/Glacier-Movement-Prediction-with-Attention-based-Recurrent-Neural-Networks-and-Satellite-Data.pdf)].  

### **Output Structure:**  
- **`/images/`**: Stores model inputs (4 consecutive patches).  
- **`/targets/`**: Stores ground truth data corresponding to each set of consecutive patches.  

---

## **3. Training Recurrent Neural Networks (RNNs)**  
The **`/DeepLearning`** folder contains scripts for training different deep learning models for glacier movement prediction.  

### **Training Process:**  
- Each model class has its own **train script**, which allows customization of hyperparameters and training settings.  
- The training process utilizes **Weights & Biases (WandB)** for real-time tracking and visualization.  
- To use WandB, create a free account and log in during training to monitor progress via a web browser.  

---

## **Contributions & Future Work**  
This repository aims to facilitate satellite-based glacier movement prediction using deep learning. Future improvements include integrating additional satellite sources and refining model architectures.  

For questions or contributions, feel free to submit an issue or pull request! 🚀  
