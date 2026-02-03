# 🔐 TRINETRA  
### *The third eye sees intent, not just input*

<p align="center">
  <img src="frontend/assets/trinetra-logo.png" alt="Trinetra Logo" width="220">
</p>

---

## 🖥️ Screenshots

### Interface Overview

<p align="left">
  <strong>1.</strong><br>
  <img src="screenshots/screen1.png" width="100%">
</p>

<p align="left">
  <strong>2.</strong><br>
  <img src="screenshots/screen2.png" width="100%">
</p>

<p align="left">
  <strong>3.</strong><br>
  <img src="screenshots/screen3.png" width="100%">
</p>

<p align="left">
  <strong>4.</strong><br>
  <img src="screenshots/screen4.png" width="100%">
</p>

Screenshots **1 to 4** showcase **TRINETRA’s cyberpunk-inspired user interface**, built for real-time AI security and intent analysis.  
The system visualizes a proactive framework that detects **prompt injection**, **data poisoning**, and **unsafe execution**.  
Scanner, logs, and service views provide clear insight into threats, system state, and security outcomes.  
Together, these screens highlight TRINETRA’s overall UI design and usability.

---

### Prompt Injection

<p align="left">
  <strong>5.</strong><br>
  <img src="screenshots/screen5.png" width="100%">
</p>

<p align="left">
  <strong>6.</strong><br>
  <img src="screenshots/screen6.png" width="100%">
</p>

Screenshots **5 and 6** illustrate the difference between **our model** and **DeepSeek**.  
When a malicious prompt is provided to DeepSeek, the model is affected and generates a response.  
In contrast, **our model detects the threat and blocks it directly at the user interface level**.

---

### Data Poisoning

<p align="left">
  <strong>7.</strong><br>
  <img src="screenshots/screen7.png" width="100%">
</p>

<p align="left">
  <strong>8.</strong><br>
  <img src="screenshots/screen8.png" width="100%">
</p>

Data poisoning can occur at **every stage of an LLM’s lifecycle**.  
Our primary focus is **real-time detection of data poisoning** during information retrieval.  
When an LLM requires additional information on a topic and searches the web, our system detects **false or misleading data** and prevents the model from being affected.

Our **URL Fetcher Agent** scans the web for **credible and authoritative sources** and provides only trusted data to the output agent.  
If no external sources are required, the prompt is sent directly to the response agent.

<p align="left">
  <strong>9.</strong><br>
  <img src="screenshots/screen9.png" width="100%">
</p>

All source URLs used during this process are **logged and tracked**, as shown in Screenshot **9**.

---

## 🚀 Overview

**TRINETRA** is an AI-native security system designed to protect Large Language Model (LLM) applications from **prompt injection**, **jailbreaks**, and **data poisoning attacks**.

Unlike traditional filters, Trinetra focuses on **intent analysis**, not just keyword matching—providing proactive, layered defense across input, execution, and output stages.

---

## ✨ Features

### 🛡️ Multi-Layered LLM Security
- Prompt injection detection (heuristics + LLM-based)
- Real-time input scanning and feedback
- Data validation sandbox for external sources
- Agentic execution monitoring (safe code/container execution)
- Output safety checks (PII, harmful, or poisoned content)

### 🔗 Credible Source Finder
- Identifies and enforces trusted, authoritative URLs
- Reduces hallucinations and misinformation

### 📊 Security Dashboard
- Scan logs and threat statistics
- Safe vs malicious input tracking
- URL classification insights

### 🔍 Preprocessing API
- Text normalization
- Image OCR
- PDF and document parsing
- QR and barcode scanning
- URL extraction and validation

### 🎨 Modern UI
- Cyberpunk-inspired dark mode
- Live diagnostics
- Interactive threat feedback

---

## 🧠 Approach to Prompt Injection & Data Poisoning

Trinetra uses a **defense-in-depth pipeline**:

1. **Instant Detection**  
   Regex and pattern-based filters block known jailbreak and injection attempts.

2. **LLM-Based Analysis**  
   Groq-powered analysis detects subtle, novel, or obfuscated attacks and assigns threat scores.

3. **Sandboxed Data Validation**  
   External data (URLs, documents) is processed in isolated agents before reaching the main model.

4. **Agentic Execution Monitoring**  
   Generated logic and code are tracked and safely executed to prevent system compromise.

5. **Output Safety Checks**  
   Final responses are scanned for PII leaks, harmful content, and poisoned data.

6. **Credible Source Enforcement**  
   Factual queries are answered only using trusted sources.

This ensures both **proactive prevention** and **reactive containment**.

---

## 🧰 Tech Stack

- **Backend:** Python, FastAPI, LangChain, Groq API, SQLite  
- **Frontend:** HTML, CSS, JavaScript  
- **Preprocessing:** pdfplumber, pytesseract, pandas, Pillow  

---
