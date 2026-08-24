import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Integrated Logistics Router API",
    description=(
        "Combines Geography, Conditions, Cost, "
        "and Payload model predictions."
    ),
    version="1.0",
)


GEOGRAPHY_URL = os.getenv(
    "GEOGRAPHY_URL",
    "http://geography-api:8000/predict",
)

CONDITIONS_URL = os.getenv(
    "CONDITIONS_URL",
    "http://conditions-api:8000/api/v1/conditions/predict",
)

COST_URL = os.getenv(
    "COST_URL",
    "http://cost-api:8000/predict",
)

PAYLOAD_URL = os.getenv(
    "PAYLOAD_URL",
    "http://telemetry-api:8000/api/v1/payload",
)


class IntegratedRouteRequest(BaseModel):
    # Geography
    order_latitude: float
    order_longitude: float
    distance_km: float = Field(..., gt=0)
    delivery_time_window_hrs: float = Field(..., ge=0)
    order_priority: float
    traffic_density_index: float = Field(..., ge=0, le=1)

    # Conditions
    weather_impact_index: float = Field(..., ge=0, le=1)
    average_speed_kmph: float = Field(..., ge=0)
    time_of_day: int = Field(..., ge=0, le=23)

    # Cost / payload
    vehicle_capacity_kg: float = Field(..., gt=0)
    order_weight_kg: float = Field(..., gt=0)
    vehicle_utilization_ratio: float = Field(..., ge=0)
    fuel_cost_per_km: float = Field(..., ge=0)
    driver_cost_per_hour: float = Field(..., ge=0)
    route_reliability_index: float = Field(..., ge=0)

    # Payload-specific derived-style inputs
    vehicle_capacity_pct: float = Field(..., ge=0, le=150)


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "integrated-logistics-router",
        "tracks": [
            "geography",
            "conditions",
            "cost",
            "payload",
        ],
    }


@app.post("/route-analysis")
async def route_analysis(data: IntegratedRouteRequest):

    geography_payload = {
        "order_latitude": data.order_latitude,
        "order_longitude": data.order_longitude,
        "distance_km": data.distance_km,
        "delivery_time_window_hrs":
            data.delivery_time_window_hrs,
        "order_priority": data.order_priority,
        "traffic_density_index":
            data.traffic_density_index,
    }

    conditions_payload = {
        "weather_impact_index":
            data.weather_impact_index,
        "average_speed_kmph":
            data.average_speed_kmph,
        "time_of_day":
            data.time_of_day,
        "traffic_density_index":
            data.traffic_density_index,
    }

    cost_payload = {
        "order_latitude": data.order_latitude,
        "order_longitude": data.order_longitude,
        "distance_km": data.distance_km,
        "delivery_time_window_hrs":
            data.delivery_time_window_hrs,
        "order_priority": data.order_priority,
        "vehicle_capacity_kg":
            data.vehicle_capacity_kg,
        "order_weight_kg":
            data.order_weight_kg,
        "vehicle_utilization_ratio":
            data.vehicle_utilization_ratio,
        "traffic_density_index":
            data.traffic_density_index,
        "average_speed_kmph":
            data.average_speed_kmph,
        "weather_impact_index":
            data.weather_impact_index,
        "time_of_day":
            data.time_of_day,
        "fuel_cost_per_km":
            data.fuel_cost_per_km,
        "driver_cost_per_hour":
            data.driver_cost_per_hour,
        "route_reliability_index":
            data.route_reliability_index,
    }

    payload_payload = {
        "cargo_weight_kg":
            data.order_weight_kg,
        "vehicle_capacity_pct":
            data.vehicle_capacity_pct,
        "delivery_deadline_hrs":
            data.delivery_time_window_hrs,
        "trip_distance_km":
            data.distance_km,
        "traffic_density_index":
            data.traffic_density_index,
        "weather_impact_index":
            data.weather_impact_index,
    }

    async with httpx.AsyncClient(
        timeout=30.0
    ) as client:

        try:
            geography_response = await client.post(
                GEOGRAPHY_URL,
                json=geography_payload,
            )

            conditions_response = await client.post(
                CONDITIONS_URL,
                json=conditions_payload,
            )

            cost_response = await client.post(
                COST_URL,
                json=cost_payload,
            )

            payload_response = await client.post(
                PAYLOAD_URL,
                json=payload_payload,
            )

        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Downstream API unavailable: {exc}",
            ) from exc

    responses = {
        "geography": geography_response,
        "conditions": conditions_response,
        "cost": cost_response,
        "payload": payload_response,
    }

    failed = {
        name: {
            "status_code": response.status_code,
            "body": response.text,
        }
        for name, response in responses.items()
        if response.status_code != 200
    }

    if failed:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "One or more track APIs failed.",
                "failures": failed,
            },
        )

    return {
        "geography": geography_response.json(),
        "conditions": conditions_response.json(),
        "cost": cost_response.json(),
        "payload": payload_response.json(),
    }