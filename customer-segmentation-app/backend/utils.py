import pandas as pd
import joblib
import pickle
import os

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predictions.csv')
CUSTOMERS_FILE = os.path.join(DATA_DIR, 'customers.csv')

# Cluster Mapping
CLUSTER_LABELS = {
    0: "Deal-Seeking Parents",
    1: "Budget-Conscious",
    2: "High-Value",
    3: "Premium Loyal"
}

def load_pkl(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)

# Load Models
scaler = load_pkl(SCALER_PATH)
model = load_pkl(MODEL_PATH)

def get_cluster_label(cluster_id):
    return CLUSTER_LABELS.get(cluster_id, "Unknown Segment")

def preprocess_and_predict(input_data: dict) -> tuple:
    # Feature columns expected by scaler in exact order:
    # ['Education', 'Marital_Status', 'Income', 'Recency', 'NumDealsPurchases', 
    #  'NumWebVisitsMonth', 'Response', 'Age', 'Total_Spend', 'Total_Purchases', 
    #  'Total_Dependents', 'Total_Campaigns_Accepted']
    
    # We provide default values for fields not consistently shown in UI
    df = pd.DataFrame([{
        'Education': getattr(input_data, 'Education', 0),
        'Marital_Status': getattr(input_data, 'Marital_Status', 0),
        'Income': input_data.Income,
        'Recency': input_data.Recency,
        'NumDealsPurchases': input_data.NumDealsPurchases,
        'NumWebVisitsMonth': input_data.NumWebVisitsMonth,
        'Response': getattr(input_data, 'Response', 0),
        'Age': input_data.Age,
        'Total_Spend': input_data.Total_Spend,
        'Total_Purchases': input_data.Total_Purchases,
        'Total_Dependents': input_data.Total_Dependents,
        'Total_Campaigns_Accepted': input_data.Total_Campaigns_Accepted
    }])
    
    scaled_data = scaler.transform(df)
    cluster = int(model.predict(scaled_data)[0])
    label = get_cluster_label(cluster)
    
    return cluster, label

def save_prediction(input_data: dict, cluster: int, label: str):
    # Prepare row
    row = {
        'Education': getattr(input_data, 'Education', 0),
        'Marital_Status': getattr(input_data, 'Marital_Status', 0),
        'Income': input_data.Income,
        'Recency': input_data.Recency,
        'NumDealsPurchases': input_data.NumDealsPurchases,
        'NumWebVisitsMonth': input_data.NumWebVisitsMonth,
        'Response': getattr(input_data, 'Response', 0),
        'Age': input_data.Age,
        'Total_Spend': input_data.Total_Spend,
        'Total_Purchases': input_data.Total_Purchases,
        'Total_Dependents': input_data.Total_Dependents,
        'Total_Campaigns_Accepted': input_data.Total_Campaigns_Accepted,
        'Cluster': cluster,
        'Cluster_Label': label
    }
    
    df = pd.DataFrame([row])
    
    # Check if predictions.csv exists
    if not os.path.isfile(PREDICTIONS_FILE):
        df.to_csv(PREDICTIONS_FILE, index=False)
    else:
        df.to_csv(PREDICTIONS_FILE, mode='a', header=False, index=False)
        
    # Also append to customers.csv so it reflects immediately in Dashboard and Explorer
    if not os.path.isfile(CUSTOMERS_FILE):
        df.to_csv(CUSTOMERS_FILE, index=False)
    else:
        df.to_csv(CUSTOMERS_FILE, mode='a', header=False, index=False)
        
def get_dashboard_stats():
    # Provide stats for the UI from customers.csv
    try:
        df = pd.read_csv(CUSTOMERS_FILE)
        total_customers = len(df)
        avg_income = int(df['Income'].mean())
        avg_spend = int(df['Total_Spend'].mean())
        
        # Segment counts
        segment_counts = df['Cluster_Label'].value_counts().to_dict()
        
        # Averages per segment
        avg_income_seg = {}
        avg_spend_seg = {}
        if not df.empty and 'Cluster_Label' in df.columns:
            avg_income_seg = df.groupby('Cluster_Label')['Income'].mean().fillna(0).round(0).to_dict()
            avg_spend_seg = df.groupby('Cluster_Label')['Total_Spend'].mean().fillna(0).round(0).to_dict()
        
        return {
            "total_customers": total_customers,
            "segments": len(segment_counts) if len(segment_counts) > 0 else 4,
            "avg_income": avg_income,
            "avg_spend": avg_spend,
            "segment_distribution": segment_counts,
            "avg_income_per_segment": avg_income_seg,
            "avg_spend_per_segment": avg_spend_seg
        }
    except Exception as e:
        print("Error in stats:", e)
        return {}

def get_customers_data():
    try:
        df = pd.read_csv(CUSTOMERS_FILE)
        return df.to_dict(orient="records")
    except Exception:
        return []

def get_predictions_data():
    if os.path.isfile(PREDICTIONS_FILE):
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            return df.to_dict(orient="records")
        except Exception:
            return []
    return []

def delete_customer(index: int):
    try:
        if not os.path.isfile(CUSTOMERS_FILE):
            return False, "Data file not found"
        df = pd.read_csv(CUSTOMERS_FILE)
        if index < 0 or index >= len(df):
            return False, "Invalid index"
        
        # Extract row
        deleted_row = df.iloc[[index]]
        # Remove row
        df.drop(index, inplace=True)
        # Save back
        df.to_csv(CUSTOMERS_FILE, index=False)
        
        # Save to deleted.csv
        deleted_file = os.path.join(DATA_DIR, 'deleted.csv')
        if not os.path.isfile(deleted_file):
            deleted_row.to_csv(deleted_file, index=False)
        else:
            deleted_row.to_csv(deleted_file, mode='a', header=False, index=False)
            
        return True, "Customer deleted successfully"
    except Exception as e:
        return False, str(e)

