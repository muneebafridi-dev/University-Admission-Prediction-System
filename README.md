# 🎓 University Admission Prediction System

An AI-powered desktop application that predicts university admission chances based on academic records. Built with Python and CustomTkinter, tailored for Pakistan's admission system across **BS**, **Master's**, and **PhD** levels.

---

##  Preview

> Run the app and select your degree level to get started.

---

##  Features

-  **Three Degree Modes** — BS, Master's, and PhD each have their own form and merit formula
-  **AI-Powered** — Trained Random Forest model for graduate-level prediction
-  **Merit Calculator** — Pakistan-standard aggregate formulas for BS and Master's
-  **Personalized Feedback** — Tells you which factor is holding your score back
-  **Dark / Light Mode** — Toggle and remembers your preference
-  **Prediction History** — Every prediction is saved automatically to a CSV log
-  **Input Validation** — Catches missing or out-of-range values before predicting

---

## 🎓 Admission Modes

| Degree | Fields Required | Formula |
|--------|----------------|---------|
| **BS** | Matric %, Intermediate %, Entry Test % | Matric 10% + Inter 40% + Entry Test 50% |
| **Master's** | CGPA, Entry Test Score, Relevant Degree | CGPA 50% + Entry Test 50% |
| **PhD** | MS/MPhil CGPA, GAT-Subject/HAT-Subject Score, Research Publications, Relevant Degree | CGPA 40% + Test 40% + Research 20% |

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/muneebafridi-dev/University-Admission-Prediction-System.git
cd University-Admission-Prediction-System
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the AI model
```bash
python train_model.py
```
This generates `model.pkl` and `scaler.pkl` — required before running the app.

### 5. Run the app
```bash
python app.py
```

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
joblib
matplotlib
customtkinter
```

---

## 📁 Project Structure

```
UniversityAdmissionPrediction/
│
├── app.py               # Main application (GUI)
├── train_model.py       # Model training script
├── dataset.csv          # Training dataset
├── model.pkl            # Trained AI model (generated)
├── scaler.pkl           # Feature scaler (generated)
├── requirements.txt     # Python dependencies
├── settings.json        # Saved theme preference (generated)
│
├── history/
│   └── predictions_log.csv   # Auto-generated prediction history
│
└── assets/              # App assets
```

---

## 🧠 How It Works

- **BS** and **Master's** use a transparent merit formula based on  standard university aggregate system
- **PhD** uses the HEC-recommended formula (CGPA + GAT-Subject/HAT-Subject + Research Publications)
- All predictions include a personalized insight identifying your weakest factor and how to improve it
- The app remembers your dark/light theme preference between sessions via `settings.json`

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Features | SOP Score, LOR Score, CGPA, Research Experience |
| Output | Admission probability (0–100%) |
| Training Data | Graduate admissions dataset |

---

## 👨‍💻 Author

**muneebafridi-dev**
- GitHub: [@muneebafridi-dev](https://github.com/muneebafridi-dev)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a **star** on GitHub — it helps others find it too!
