####### Brain Tumor Detection using Otsu Thresholding with Genetic Algorithm Optimization
### 1. Setup Library
*   **Libraries Used**:
    *   `numpy`: For numerical operations.
    *   `opencv-python` (`cv2`): For image processing functions (loading images, calculating histograms, traditional Otsu).
    *   `matplotlib` & `seaborn`: For plotting and visualizing the analysis results.
    *   `google.colab`: For file handling within the Colab environment.

### 2. Dataset Setup

The script provides two options for loading your image dataset [1]:

*   **Option 1: Upload a ZIP file**
    1.  When prompted, select option `1`.
    2.  Use the file upload interface to upload a `.zip` file containing your dataset.
    3.  The script will automatically extract the contents into a `dataset/` directory.

*   **Option 2: Use Google Drive**
    1.  When prompted, select option `2`.
    2.  Authorize the notebook to access and mount your Google Drive.
    3.  Provide the full path to your dataset folder within Google Drive (e.g., `/content/drive/MyDrive/your_dataset_folder`).

### 3. Running the Experiment

1.  Open the `.ipynb` file in Google Colaboratory.
2.  Run the notebook cells sequentially.
3.  Follow the prompts to load the dataset using one of the methods described above.
4.  When asked, provide the full path to a specific image you want to analyze (e.g., `/content/dataset/yes/Y1.jpg`).
5.  The script will execute the GA optimization process and display a comprehensive analysis plot comparing the GA-Otsu method with the traditional Otsu method.
6.  You will have the option to save the final analysis image to your Colab environment.
