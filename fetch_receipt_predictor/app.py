from fastapi import FastAPI
from pydantic.fields import Field
from fetch_receipt_predictor.inference import  predict_receipts
from fetch_receipt_predictor.model import  train_model
from pydantic import BaseModel, conint, conlist, model_validator 

app = FastAPI()

model, scaler_X, scaler_y = train_model(data_path="data_daily.csv", model_save_path="model.pth")

class ReceiptPredictionRequest(BaseModel):
    month: list[conint(ge=1, le=12)] 
    year: list[conint(ge=2021)]

    @model_validator(mode="after")
    def validate_lengths(self):
        if len(self.month) != len(self.year):
            raise ValueError("The number of months and years must be the same")
        return self

class ReceiptPredictionResponse(BaseModel):
    month: int
    year: int
    predicted_receipts: float

@app.post("/predict/")
async def predict(request: ReceiptPredictionRequest) -> list[ReceiptPredictionResponse]:
    start_year = 2021
    months_list = []
    for year, month in zip(request.year, request.month):
        month_number = (year - start_year) * 12 + month
        months_list.append(month_number)
    predictions = predict_receipts(model, scaler_X, scaler_y, months_list)


    response_data = []
    for year, month, prediction in zip(request.year, request.month, predictions):
        response_data.append(ReceiptPredictionResponse(
            month=month,
            year=year,
            predicted_receipts=prediction
        ))

    return response_data
