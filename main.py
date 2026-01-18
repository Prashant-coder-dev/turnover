from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API URLs
NEPSELYTICS_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/homepage"
NEPSE_TURNOVER_URL = "https://tms59.nepsetms.com.np/tmsapi/rtApi/admin/vCache/marketTurnover"

# Root endpoint
@app.get("/")
def root():
    return {"status": "NEPSE Data API Running"}

# -------------------------------
# Nepselytics Homepage Data
# -------------------------------
@app.get("/homepage-data")
async def homepage_data():
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(NEPSELYTICS_URL)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch homepage market data"
        )

    return resp.json()

# -------------------------------
# Market Turnover (NEPSE Proxy)
# -------------------------------
@app.get("/market-turnover")
async def market_turnover():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(NEPSE_TURNOVER_URL, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch NEPSE market turnover"
        )

    return resp.json()

