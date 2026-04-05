# Customer Segmentation App

A full-stack web application designed to segment customers into predefined clusters using an ML model (K-Means Clustering). 

## 📁 Project Structure

```text
customer-segmentation-app/
│
├── backend/
│   ├── main.py        # FastAPI app routes and configuration
│   ├── utils.py       # Helper functions (preprocessing, model loading, saving)
│   ├── model/         # Pre-trained ML models
│   │   ├── kmeans_model.pkl
│   │   └── scaler.pkl
│   ├── data/          # Original dataset and predictions storage
│   │   ├── customers.csv
│   │   └── predictions.csv
│
├── frontend/
│   ├── index.html     # UI layout
│   ├── styles.css     # Clean modern dashboard styling
│   └── script.js      # Form processing, charting (Chart.js), API fetching
│
├── requirements.txt   # Python environment dependencies
└── README.md          # Project instructions
```

## 🚀 Features

- **Overview Dashboard**: Metrics and visualizations (Pie & Bar charts) for customer clustering data.
- **Predict Customer Segment**: Input customer data dynamically into a form to fetch their cluster and label.
- **Segment Profiles**: Detailed business analytics and strategies for each segment (Premium Loyal, Budget-Conscious, High-Value, Deal-Seeking Parents).
- **Data Explorer**: Live table of all customers dynamically fetched from the FastAPI backend. You can search or filter through the results.

## 🛠️ Setup Instructions

### 1. Backend Setup

Create a virtual environment and launch the FastAPI server.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt

# Run the FastAPI app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Untill you deploy The API will be available at `http://localhost:8000`*

### 2. Frontend Setup

The frontend uses standard HTML/CSS/JS with no build step.

Options:

1. Open directly:

```bash
open frontend/index.html
```

2. Run a local static server (recommended):

```bash
cd frontend
python3 -m http.server 5500
```

Then open:

- `http://localhost:5500` in your browser

3. If using VS Code Live Server, open `frontend/index.html` and click "Go Live".

The app will call the backend API at `http://127.0.0.1:8000`.

## 📄 API Endpoints

- `POST /predict` - Accepts patient features and returns Predicted Segment.
- `GET /data` - Returns paginated/processed base customers list.
- `GET /predictions` - Returns previously saved dynamic predictions.
- `GET /stats` - Returns overall descriptive cluster stats.
