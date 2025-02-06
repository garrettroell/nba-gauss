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

@app.post("/predict")
def predict_gp(data: PredictionRequest):
    print("Received Data:", data)  # Debugging print statement

    # Convert input data to numpy arrays (ensuring correct dtype)
    x_train = np.array(data.x_values, dtype=np.float64).reshape(-1, 1)  # Shape: (n_samples, 1)
    y_train = np.array(data.y_values, dtype=np.float64)

    # Ensure the periodicity is always longer than 5
    # periodicity_value = max(len(x_train) / 2, 6)
    quarter_lenth = len(x_train) // 4
    full_length = len(x_train)

    # Define a modified Gaussian Process Kernel with even smoother properties:
    kernel = (
        ConstantKernel(1.5, (0.5, 100))
        * (
        #     # Increase length scale to produce smoother functions:
            RBF(length_scale=quarter_lenth, length_scale_bounds=(quarter_lenth, full_length))
            + Matern(length_scale=quarter_lenth, nu=2.5, length_scale_bounds=(quarter_lenth, full_length))
            + RationalQuadratic(length_scale=quarter_lenth, alpha=1.0, length_scale_bounds=(quarter_lenth, full_length))
        )
        # Increase the white noise level for additional smoothing
        + WhiteKernel(noise_level=2.0, noise_level_bounds=(0.1, 5))
    )

    # Initialize and fit GP model with normalization for better scaling
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=20, normalize_y=True)
    gp.fit(x_train, y_train)

    # Create an interpolation range for prediction
    x_min = 0
    x_max = np.max(x_train) + 1
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)

    # Predict mean and uncertainty over the test range
    y_mean, y_std = gp.predict(x_range, return_std=True)

    # Predict value at requested `x_target`
    x_target_array = np.array([[data.x_target]], dtype=np.float64)
    y_target_pred, target_sigma = gp.predict(x_target_array, return_std=True)

    # Return results
    return {
        "x_values": x_range.flatten().tolist(),
        "y_values": y_mean.tolist(),
        "y_std_values": y_std.tolist(),
        "x_target":  data.x_target,
        "y_target": y_target_pred[0],
        "y_std_target": target_sigma[0],
    }



# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3755)
