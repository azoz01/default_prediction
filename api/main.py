from typing import Union
from fastapi import FastAPI
from api.data_model import ClientData, ClientDataList
from api.service import DefaultApiService

default_api_service = DefaultApiService()
app = FastAPI()


@app.post("/")
async def root(data: Union[ClientData, ClientDataList]):
    data_to_predict = default_api_service.extract_data_from_body(data)
    converted_frame = default_api_service.convert_json_data_to_pandas(
        data_to_predict
    )
    prediction = default_api_service.make_prediction(data=converted_frame)
    print(prediction)
    return prediction

