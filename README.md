🤖 **AI Resume Screening System**

An end-to-end NLP & Machine Learning project that automatically analyzes resumes and predicts the most relevant job role.
The application also extracts skills and years of experience from resumes and provides results through a Streamlit web app.

📌 **Project Overview**

Recruiters receive hundreds of resumes daily. Manually screening them is time-consuming and inefficient.

This project automates the resume screening process by:

Classifying resumes into job roles

Extracting technical skills

Extracting years of experience

Providing an interactive web interface for real-time predictions

Finally user can download the excel sheet

🎯 **Features**

✔ Resume Classification (ML Model)
✔ Skills Extraction (NLP Keyword Matching)
✔ Experience Extraction (Regex Based)
✔ Supports multiple formats: PDF, DOCX, DOC
✔ Interactive Streamlit Web App
✔ End-to-end ML pipeline

🧠 **Machine Learning Workflow**
1️⃣ **Data Collection**

**Dataset contains resumes categorized into:**

React Developer

SQL Developer

Workday

Peoplesoft

2️⃣ **Text Extraction**

**Resumes are parsed using:**

File Type	Library
PDF	PyPDF2
DOCX	python-docx
DOC	**LibreOffice** conversion

3️⃣ **Text Preprocessing**

• Lowercasing
• Stopword removal
• Tokenization
• Lemmatization
• Special token replacement for numbers

4️⃣ **Feature Engineering**

Text converted to numerical vectors using:

TF-IDF Vectorization

Unigrams + Bigrams

Feature selection applied

5️⃣ **Model Building**

**Five algorithms were compared:**

**Model	Result**
Naive Bayes	100% Accuracy
Logistic Regression	100% Accuracy
**Support Vector Machine (Final Model)	⭐ Best**
Random Forest	100% Accuracy
KNN	Slightly lower
Final Model Selected:

Linear **SVM** with Hyperparameter Tuning

6️⃣ **Model Validation**

• Stratified Train-Test Split
• 10-Fold Cross Validation
• Repeated Cross Validation

**Final Mean CV Accuracy ≈ 98%**

🧾 **Additional NLP Features
🔹 Skills Extraction**

Uses a domain-specific skills dictionary covering:

Web Development

Database

DevOps Tools

Workday & Peoplesoft skills

🔹 **Experience Extraction**

Regex based detection of patterns like:

“3 years experience”

“5+ yrs”

“2 year experience”

🌐 **Streamlit Web Application**

Users can upload resumes and get:

Predicted Job Role(Resumes are sorted Category wise)

Extracted Skills

Years of Experience(Sorted in ascending order for easy shortlisting)

🛠 **Tech Stack**

**Python**

Scikit-learn

NLTK

TF-IDF

Regex

Streamlit

LibreOffice



Install LibreOffice (Required for DOC files)

**Download and install LibreOffice from:**
https://www.libreoffice.org/



**Author: Er. Pratiksha Mhaske** 

**LinkedIn:** https://www.linkedin.com/in/pratiksha-mhaske

**GitHub:** https://github.com/PratikshaMhaske
