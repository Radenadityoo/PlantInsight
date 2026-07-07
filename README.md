# PlantInsight

PlantInsight is a Flask web app for citrus leaf disease classification. It compares two CNN models, ResNet50 and MobileNetV2, and shows prediction results, history, and a compact dataset summary on the About page.

## Features

- Upload a citrus leaf image and get predictions from both models
- Compare confidence scores and display the best model
- Save prediction history in the database
- Show a dataset distribution chart from a generated summary file
- Deployable on Render with PostgreSQL

## Tech Stack

- Python 3
- Flask
- TensorFlow / Keras
- Flask-SQLAlchemy
- PostgreSQL on Render
- Gunicorn for production

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python run.py
```

4. Open the app in your browser at `http://127.0.0.1:5000`.

## Render Deployment

This repository includes a `render.yaml` blueprint. Render uses:

- `gunicorn run:app --bind 0.0.0.0:$PORT`
- a PostgreSQL database created from the blueprint

## Notes

- The model files are stored with Git LFS.
- The About page reads from `app/data/dataset_summary.json` instead of scanning the full dataset folder at runtime.
- If you clone this repo, make sure Git LFS is installed before pulling the model files.
