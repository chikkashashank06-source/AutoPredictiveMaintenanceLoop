# 🚗 AutoPredictiveMaintenanceLoop

Agentic AI–based system for autonomous predictive vehicle maintenance and proactive service scheduling.

## 🔍 Overview
This project analyzes telematics and historical vehicle data to:
- Predict component failures
- Automatically schedule maintenance
- Engage customers proactively
- Optimize service operations
- Provide closed-loop manufacturing feedback
- Monitor agent behavior (UEBA)

## 🧠 Architecture
- **Frontend**: Streamlit (Agentic Control Center UI)
- **ML Model**: Random Forest (failure prediction)
- **AI Agents**:
  - Telemetry Agent
  - Prediction Agent
  - Scheduling Agent
  - Manufacturing Feedback Agent
  - Security (UEBA) Agent

## ⚙️ How to Run
```bash
pip install streamlit scikit-learn pandas numpy
streamlit run g.py
