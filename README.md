# DermaCareAI Python Website

Exam-friendly version using Python Flask for backend, HTML templates for structure, CSS for design, and `backend/model.h5` for prediction.

Project layout:

- `backend/app.py` - Python Flask backend
- `backend/model.h5` - AI model
- `templates/` - HTML pages
- `static/` - CSS and images
- `uploads/` - uploaded scan images
- `scan_history.json` - admin scan history

Run:

```powershell
cd backend
..\venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`.
