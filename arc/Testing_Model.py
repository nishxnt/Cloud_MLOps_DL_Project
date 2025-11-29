import base64
from google.cloud import aiplatform

# ---- FILL THESE ----
PROJECT_ID = "mlops-project-479512"          # or your actual project id
LOCATION = "europe-west3"               # your region
ENDPOINT_ID = "6025044444259024896"   # copy from Vertex endpoint page

endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"

# ---- 1) Encode local image as base64 (same as you did) ----
image_path = "10212.jpg"  # make sure this exists in the Workbench VM

with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

instance = {
    "data": {
        "b64": b64
    }
}
instances = [instance]

# ---- 2) Create client and call endpoint ----
client = aiplatform.gapic.PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)

response = client.predict(
    endpoint=endpoint_name,
    instances=instances,
    parameters=None,
)

print("Raw response:", response)
print("Predictions:", response.predictions)