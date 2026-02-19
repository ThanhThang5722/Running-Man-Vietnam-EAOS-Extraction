from fastapi import FastAPI

from schemas import (
    ModelInput,
    ModelOutput
)

from services import EAOSModelService

app = FastAPI()

mlService = EAOSModelService()

@app.post("/predict")
async def predict(input_data: ModelInput):

    raw_results = mlService.predict(input_data.text)

    return ModelOutput(results=raw_results)