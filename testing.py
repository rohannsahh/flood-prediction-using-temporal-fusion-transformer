import pandas as pd
import numpy as np

# Parameters
start_date = "2020-01-01"
num_days = 160
latitude = 9.5
longitude = 77.25
state = "Kerala"
district = "Idukki"

# Generate date range
dates = pd.date_range(start=start_date, periods=num_days, freq="D")

# Generate dummy data
data = {
    "TIME": dates,
    "LATITUDE": [latitude] * num_days,
    "LONGITUDE": [longitude] * num_days,
    "STATE": [state] * num_days,
    "DISTRICT": [district] * num_days,
    "RAINFALL": np.random.uniform(0, 20, num_days),
    # "FLOOD": np.random.choice([0, 1], size=num_days, p=[0.9, 0.1])  # 10% floods
}

df = pd.DataFrame(data)


# Optional: Save to CSV
df.to_csv("idukki_rainfall_dataset.csv", index=False)

print(df.head())
