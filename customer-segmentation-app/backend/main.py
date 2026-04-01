from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import utils
import uvicorn

app = FastAPI(title="Customer Segmentation Dashboard API")

# Enable CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerInput(BaseModel):
    Education: Optional[int] = 0
    Marital_Status: Optional[int] = 0
    Income: float
    Recency: int
    NumDealsPurchases: int
    NumWebVisitsMonth: int
    Response: Optional[int] = 0
    Age: int
    Total_Spend: int
    Total_Purchases: int
    Total_Dependents: int
    Total_Campaigns_Accepted: int

@app.get("/")
def read_root():
    return {"message": "Customer Segmentation API is running."}

@app.post("/predict")
def predict_segment(data: CustomerInput):
    try:
        cluster, label = utils.preprocess_and_predict(data)
        utils.save_prediction(data, cluster, label)
        return {
            "cluster": cluster,
            "segment": label,
            "message": "Prediction successful and stored."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/data")
def get_data():
    try:
        data = utils.get_customers_data()
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions():
    try:
        data = utils.get_predictions_data()
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_dashboard_stats():
    try:
        stats = utils.get_dashboard_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{index}")
def delete_customer_endpoint(index: int):
    success, message = utils.delete_customer(index)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
