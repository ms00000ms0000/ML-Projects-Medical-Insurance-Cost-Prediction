# 🏥 Medical Insurance Cost Prediction — Machine Learning Model  

## 🚀 Overview  
This **Machine Learning project** focuses on **Medical Insurance Cost Prediction** using the **Linear Regression** algorithm.  
The model predicts **insurance charges** based on input features such as **Age, Sex, BMI, Number of Children, Smoking Status, and Region**.  

Comprehensive **Exploratory Data Analysis (EDA)** was conducted, including **Age Distribution, Sex Distribution, BMI Distribution, and Charges Distribution**, to identify key factors influencing medical costs.  
The model achieved an **accuracy of 75%**, providing reliable estimates of insurance expenses.  

This project demonstrates how **Machine Learning** can be leveraged in **healthcare analytics** to assist **insurance providers** and **individuals** in understanding **cost patterns** and **risk factors**.  

---

## 📘 About the Project  
This project applies **supervised learning regression techniques** to predict **medical insurance costs** using demographic and health-related attributes.  
The **Linear Regression** algorithm was chosen for its simplicity and interpretability, allowing for clear insights into how each factor influences the final insurance charges.  

It is a practical example of **predictive analytics in healthcare**, providing useful insights into cost determinants and risk estimation.  

---

## 🧠 Model Architecture  
The project follows a structured machine learning workflow consisting of:  

1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables (Sex, Smoker, Region)  
   - Normalizing numerical data  

2. **Model Training**  
   - Applied **Linear Regression** algorithm  

3. **Evaluation Metrics**  
   - R² Score  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  

---

## 🧾 Dataset Description  
The dataset contains demographic, lifestyle, and regional attributes affecting medical insurance charges.  
It enables the prediction of healthcare costs for individuals based on personal and lifestyle factors.  

| Feature | Description |
| :------- | :-------------------------------------------------------------- |
| `age` | Age of the individual |
| `sex` | Gender of the individual (male/female) |
| `bmi` | Body Mass Index (measure of body fat) |
| `children` | Number of children/dependents covered by insurance |
| `smoker` | Smoking status (yes/no) |
| `region` | Residential area (northeast, northwest, southeast, southwest) |
| `charges` | Target variable — medical insurance cost |

---

## ⚙️ Tech Stack & Libraries  

**Programming Language:**  
- Python 🐍  

**Libraries Used:**  
- **NumPy** — Numerical computations  
- **Pandas** — Data manipulation and preprocessing  
- **Scikit-learn** — Model building and evaluation  
- **Matplotlib / Seaborn** — Data visualization and EDA  

---

## 🔍 Features  
- Predicts **medical insurance costs** based on demographic and health data  
- Performs **extensive EDA** to understand influential factors  
- Uses **Linear Regression** for interpretable predictions  
- Evaluates using multiple error metrics  
- Applicable in **insurance risk assessment** and **cost estimation**  

---

## 📊 Results  
| Metric | Score |
| :------ | :----- |
| **Accuracy (R²)** | 75% |
| **MAE** | Low |
| **MSE** | Moderate |
| **Model Type** | Linear Regression |

The model effectively predicts insurance costs, identifying **smoking status**, **age**, and **BMI** as the most significant factors affecting charges.  

---

## 📁 Repository Structure  

```
📦 ML_Project_Medical_Insurance_Cost_Prediction
│
├── medical_insurance_cost_prediction.ipynb # Jupyter Notebook implementation
├── insurance.csv # Dataset used
└── README.md # Documentation file
```

---

## 🧪 How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/ms00000ms0000/ML-Project-Medical-Insurance-Cost-Prediction.git
   cd ML-Project-Medical-Insurance-Cost-Prediction


2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook medical_insurance_cost_prediction.ipynb
   ```

4. **Execute all cells to train, test, and evaluate the model.**

---

## 📈 Future Improvements

* Implement Polynomial Regression or Random Forest Regressor for higher accuracy

* Integrate Streamlit UI for real-time cost prediction

* Add feature scaling and hyperparameter tuning for better performance

* Deploy the model using Flask or FastAPI

---

## 👨‍💻 Developer

Developed by: Mayank Srivastava
