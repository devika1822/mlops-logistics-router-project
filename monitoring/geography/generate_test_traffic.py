import random
import requests


API_URL = "http://localhost:8001/predict"

NUMBER_OF_REQUESTS = 20

random.seed(42)


for request_number in range(1, NUMBER_OF_REQUESTS + 1):

    payload = {
        "order_latitude": random.uniform(12.8, 13.2),
        "order_longitude": random.uniform(77.4, 77.8),
        "distance_km": random.uniform(10, 100),
        "delivery_time_window_hrs": random.uniform(5, 20),
        "order_priority": random.randint(1, 5),
        "traffic_density_index": random.uniform(0.2, 0.9),
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            prediction = response.json()

            print(
                f"Request {request_number:02d}: "
                f"distance={payload['distance_km']:.2f}, "
                f"traffic={payload['traffic_density_index']:.2f} "
                f"-> {prediction}"
            )

        else:
            print(
                f"Request {request_number:02d} failed: "
                f"HTTP {response.status_code} "
                f"{response.text}"
            )

    except requests.RequestException as error:
        print(
            f"Request {request_number:02d} failed: {error}"
        )