from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time

app = FastAPI(title="NEPSE Unified Market Data API")

# -------------------------------------------------
# CORS settings
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# API URLs
# -------------------------------------------------
NEPSELYTICS_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/homepage"
NEPSE_TURNOVER_URL = "https://tms59.nepsetms.com.np/tmsapi/rtApi/admin/vCache/marketTurnover"
NEPSELYTICS_FLOORSHEET_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/floorsheet"
NEPALIPAISA_INDEX_URL = "https://nepalipaisa.com/api/GetIndexLive"

# -------------------------------------------------
# Root Endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "NEPSE Data API Running",
        "endpoints": [
            "/homepage-data",
            "/market-turnover",
            "/index-live",
            "/floorsheet",
            "/floorsheet/totals"
        ]
    }

# -------------------------------------------------
# Nepselytics Homepage Data
# -------------------------------------------------
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

# -------------------------------------------------
# Market Turnover (NEPSE Proxy)
# -------------------------------------------------
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

# -------------------------------------------------
# NEPSE Index Live (NepaliPaisa)
# -------------------------------------------------
@app.get("/index-live")
async def index_live():
    """
    Live NEPSE & sub-index data from NepaliPaisa
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://nepalipaisa.com"
    }

    params = {
        "_": int(time.time() * 1000)  # cache-buster
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            NEPALIPAISA_INDEX_URL,
            headers=headers,
            params=params
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch NEPSE index live data"
        )

    return resp.json()

# -------------------------------------------------
# Floorsheet Data (Live Trading)
# -------------------------------------------------
@app.get("/floorsheet")
async def floorsheet(
    page: int = Query(0, ge=0, description="Page number (0-indexed)"),
    size: int = Query(500, ge=1, le=500, description="Number of records per page (max 500)"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }

    if size > 100:
        all_records = []
        pages_needed = (size + 99) // 100  # ceiling division

        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(pages_needed):
                current_page = page + i
                params = {
                    "page": current_page,
                    "Size": 100,
                    "order": order
                }

                resp = await client.get(
                    NEPSELYTICS_FLOORSHEET_URL,
                    params=params,
                    headers=headers
                )

                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Failed to fetch floorsheet data (page {current_page})"
                    )

                payload = resp.json()
                records = payload.get("data", {}).get("content", [])

                all_records.extend(records)

                if len(records) < 100:
                    break

        all_records = all_records[:size]

        return {
            "success": True,
            "message": f"Fetched {len(all_records)} records",
            "data": all_records,
            "pagination": {
                "page": page,
                "size": len(all_records),
                "requested_size": size
            }
        }

    else:
        params = {
            "page": page,
            "Size": size,
            "order": order
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                NEPSELYTICS_FLOORSHEET_URL,
                params=params,
                headers=headers
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail="Failed to fetch floorsheet data"
            )

        return resp.json()

# -------------------------------------------------
# Floorsheet Totals
# -------------------------------------------------
@app.get("/floorsheet/totals")
async def floorsheet_totals(
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            NEPSELYTICS_FLOORSHEET_URL,
            params={"page": 0, "Size": 1, "order": order},
            headers=headers
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch floorsheet totals"
        )

    payload = resp.json()
    data = payload.get("data", {})

    return {
        "success": True,
        "data": {
            "totalAmount": data.get("totalAmount", 0),
            "totalQty": data.get("totalQty", 0),
            "totalTrades": data.get("totalTrades", 0)
        }
    }
