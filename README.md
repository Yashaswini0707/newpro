### README for **Marine AI Dashboard: JALA JEEVA**  

---

#### **Overview**  
The **Marine AI Dashboard** application, named **JALA JEEVA**, leverages AI to monitor marine ecosystems by detecting plastic pollution in ocean images. It uses a pre-trained MobileNetV2 model to classify images into "Clean Ocean" or "Polluted Ocean." Additionally, it provides various features like prediction history, webcam-based predictions, heatmap analysis, and a business idea submission platform for ocean conservation efforts.

---

### **Features**  

1. **Home Page**:  
   - Displays a welcoming title and project description.  
   - Animated rolling tips to guide users for optimal use.

2. **Prediction**:  
   - Upload an image (jpg, png, jpeg) for prediction.  
   - Displays the predicted category (Clean/Polluted) with confidence percentage.  
   - Option to save predictions to a CSV file.

3. **Webcam Integration**:  
   - Captures images using the webcam for real-time predictions.  
   - Displays the captured image and prediction results.  
   - Saves predictions automatically.

4. **Prediction History**:  
   - Displays previously saved predictions from a CSV file.  
   - Option to download the prediction history as a CSV file.

5. **Graphs with Map**:  
   - Displays a heatmap using Folium to show pollution intensity data across global locations.  
   - Interactive map visualization for enhanced user experience.

6. **Business Idea Submission**:  
   - Form to submit company details for ocean conservation projects.  
   - Saves the company data to a CSV file (`company_partners.csv`).

7. **Marine Ecosystem VR/AR**:  
   - Embedded virtual reality experience to explore marine ecosystems.  

---

### **Installation and Setup**  

1. **Environment Setup**:  
   Ensure Python 3.7+ is installed.  
   Install required libraries by running:
   ```bash
   pip install streamlit tensorflow opencv-python-headless pandas folium pillow streamlit-folium
   ```

2. **Running the App**:  
   Save the script as `app.py` and run the following command:
   ```bash
   streamlit run app.py
   ```

3. **Background Image and Models**:  
   - Ensure the background image is accessible via the provided URL or modify it in the `set_background()` function.  
   - Ensure `marine_ai_model.h5` is available or the model will be created and saved locally.

---

### **Folder Structure**  

```
/project-folder
│
├── app.py                    # Main Streamlit application
├── marine_ai_model.h5        # Trained model file (optional, will be generated)
├── predictions.csv           # CSV file to store predictions (auto-created)
├── company_partners.csv      # CSV for storing business submissions (auto-created)
└── requirements.txt          # Python dependencies (optional)
```

---

### **Usage Instructions**  

1. **Upload Image**:  
   Navigate to the **Prediction** page to upload an image for pollution detection.
   
2. **Use Webcam**:  
   On the **Webcam** page, capture a live image for real-time predictions.

3. **View History**:  
   Visit the **History** page to view or download past predictions.

4. **Submit Business Idea**:  
   Go to the **Business Idea** page, fill in the form, and submit details for ocean conservation projects.

5. **Explore VR/AR**:  
   Navigate to **Marine Ecosystem VR/AR** to explore marine life virtually.

---

### **Customization**  

- **Background Image**:  
  Modify the URL in the `set_background()` function to change the background.
  
- **Prediction Classes**:  
  Adjust the `classes` array in `predict_image()` to modify the labels used.

- **Tips**:  
  Update the `TIPS` list to add or change the rolling tips displayed on the app.

---

### **Dependencies**  

- `streamlit`  
- `tensorflow`  
- `opencv-python-headless`  
- `pandas`  
- `folium`  
- `PIL` (Pillow)  
- `streamlit-folium`

---

### **Contributions**  

Feel free to contribute to the project by forking the repository and submitting pull requests for improvements or bug fixes.

---
