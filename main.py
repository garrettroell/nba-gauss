from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, RBF, WhiteKernel, 
                                              Matern, ExpSineSquared, RationalQuadratic)
import uvicorn

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model (ensuring float inputs)
class PredictionRequest(BaseModel):
    x_values: list[float]  # List of X values
    y_values: list[float]  # Corresponding Y values
    x_target: float  # Single X value to predict

# Test endpoint to check if the server is running
@app.get("/")
def read_root():
    return "Welcome to Garrett's Gaussian Process server!"

# Define Gaussian Process prediction endpoint
@app.post("/predict")
def predict_gp(data: PredictionRequest):
    print("Received Data:", data)  # Debugging print statement

    # Convert input data to numpy arrays (ensuring correct dtype)
    x_train = np.array(data.x_values, dtype=np.float64).reshape(-1, 1)  # Ensure shape (n_samples, 1)
    y_train = np.array(data.y_values, dtype=np.float64)  # Ensure Y is float64

    # Define Gaussian Process Kernel
    kernel = (
        ConstantKernel(1.5, (0.5, 100)) *  
        (RBF(length_scale=7, length_scale_bounds=(3, 20)) +  
         Matern(length_scale=5, nu=1.5) +  
         ExpSineSquared(length_scale=3, periodicity=len(x_train) / 2) +  
         RationalQuadratic(length_scale=5, alpha=1.0))  
        + WhiteKernel(noise_level=0.8, noise_level_bounds=(0.1, 5))
    )

    # Initialize and fit GP model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=20)
    gp.fit(x_train, y_train)

    # Ensure `x_range` is interpolated exactly like frontend
    x_min = 0
    x_max = np.max(x_train) + 1
    x_range = np.array([x_min + (i * (x_max - x_min)) / 99 for i in range(100)]).reshape(-1, 1)

    # Predict mean and uncertainty over the test range
    y_mean, y_std = gp.predict(x_range, return_std=True)

    # Predict value at requested `x_target`
    x_target_array = np.array([[data.x_target]], dtype=np.float64)
    y_target_pred, target_sigma = gp.predict(x_target_array, return_std=True)

    # Predict value at the last observed X for comparison
    x_last = np.array([[np.max(x_train)]], dtype=np.float64)
    y_last_pred, last_sigma = gp.predict(x_last, return_std=True)

    # Return results
    return {
        "x_range": x_range.flatten().tolist(),  # Matches frontend interpolation
        "y_mean": y_mean.tolist(),
        "y_uncertainty": y_std.tolist(),
        "target_prediction": {
            "x": data.x_target,
            "y": y_target_pred[0],
            "uncertainty": 2 * target_sigma[0]
        },
        "last_prediction": {
            "x": float(np.max(x_train)),
            "y": y_last_pred[0],
            "uncertainty": 2 * last_sigma[0]
        }
    }

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3755)
