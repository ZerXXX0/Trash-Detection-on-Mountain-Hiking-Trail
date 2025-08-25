# ⛰️ Trash Detection on Mountain Hiking Trail 🗑️

> This project provides a complete workflow for detecting trash on mountain hiking trails using the **YOLOv12** object detection model. The goal is to create a tool that can help automate the process of identifying and locating litter, contributing to cleaner natural environments.

---

### 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ZerXXX0/Trash-Detection-on-Mountain-Hiking-Trail.git](https://github.com/ZerXXX0/Trash-Detection-on-Mountain-Hiking-Trail.git)
    ```

2.  **Install Dependencies**
    Navigate into the project directory and install the required Python libraries.
    ```bash
    cd Trash-Detection-on-Mountain-Hiking-Trail
    pip install -r requirements.txt
    ```

3.  **Run the Detection Script**
    Use the `deploy.py` script to run inference on an image. Make sure the trained model is available in the `/model` directory.
    ```bash
    streamlit run deploy.py
    ```

---

### 💻 Project Components

* **Model (`/model`)**: Contains the trained YOLOv12 model weights used for trash detection. Model size used for deploy.py is S, to prevent performance issue on streamlit cloud. Another model is available on kaggle notebook output.
* **Notebooks**:
    * `yolov12-trash-on-mountain.ipynb`: The complete process of training the YOLOv12 model on a custom dataset of mountain trail trash.
    * `testing-mountain-trash.ipynb`: A notebook for evaluating the trained model's performance on a test set of images.
* **Deployment (`deploy.py`)**: A simple Python script to load the model and perform detection on a single image.

---

### 🤝 Contributing & License

* **License**: This project is licensed under the **MIT License**.
* **Contributing**: Contributions are welcome! Please feel free to fork the repository and submit a pull request with your improvements.
