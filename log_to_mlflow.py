import mlflow.onnx
import onnx

# Load the ONNX model from the file
onnx_model = onnx.load("bestmlflow.onnx")

# Start the MLflow run and log the model
with mlflow.start_run():
    mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="onnx-model")
