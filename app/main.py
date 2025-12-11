from fastapi import FastAPI
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.storage_service import download_blob, upload_segmented_image
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "brain-tumor-ml",
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    blob_url = payload.blob_url

    image_bytes = download_blob(blob_url)


    segmented_bytes = image_bytes  
    has_tumor = True
    confidence = 0.95
    findings = ["Mock finding"]
    recommendation = ["Mock recommendation"]
    summary = "Mock summary"
    metadata = {"device": "cpu", "inference_time_ms": 0}

    segmented_image_url = upload_segmented_image(segmented_bytes, blob_url)


    return AnalyzeResponse(
        has_tumor=has_tumor,
        confidence=confidence,
        segmented_image_url=segmented_image_url,
        summary=summary,
        findings=findings,
        recommendations=recommendation,
        metadata=metadata
)

