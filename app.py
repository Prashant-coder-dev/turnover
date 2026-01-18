from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEPSE_API_URL = "https://tms59.nepsetms.com.np/tmsapi/rtApi/admin/vCache/marketTurnover"

@app.get("/")
def root():
    return {"status": "NEPSE Proxy Running"}

@app.get("/market-turnover")
async def market_turnover():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(NEPSE_API_URL, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch NEPSE data"
        )

    return resp.json()
