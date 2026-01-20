from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math

app = FastAPI(title="NEPSE Data API")

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# External API URLs
# --------------------------------------------------
NEPSELYTICS_HOME_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/homepage"
NEPSE_TURNOVER_URL = "https://tms59.nepsetms.com.np/tmsapi/rtApi/admin/vCache/marketTurnover"
NEPSELYTICS_FLOORSHEET_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/floorsheet"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

# --------------------------------------------------
# Root
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "NEPSE Data API Running",
        "endpoints": [
            "/homepage-data",
            "/market-turnover",
            "/floorsheet"
        ]
    }

# --------------------------------------------------
# Nepselytics Homepage
# --------------------------------------------------
@app.get("/homepage-data")
async def homepage_data():
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(NEPSELYTICS_HOME_URL, headers=HEADERS)

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "Failed to fetch homepage data")

    return resp.json()

# --------------------------------------------------
# Market Turnover
# --------------------------------------------------
@app.get("/market-turnover")
async def market_turnover():
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(NEPSE_TURNOVER_URL, headers=HEADERS)

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "Failed to fetch market turnover")

    return resp.json()

# --------------------------------------------------
# Floorsheet (Merged Pagination Logic)
# --------------------------------------------------
@app.get("/floorsheet")
async def floorsheet(
    page: int = Query(0, ge=0, description="Page number (0-indexed)"),
    size: int = Query(500, ge=1, le=500, description="Records per request (max 500)"),
    order: str = Query("desc", regex="^(asc|desc)$")
):
    """
    Floorsheet proxy with merged pagination.

    Nepselytics internally limits data to ~100 rows per request.
    This endpoint transparently fetches multiple pages and merges them.
    """

    PAGE_LIMIT = 100
    pages_required = math.ceil(size / PAGE_LIMIT)
    all_records = []

    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(pages_required):
            current_page = page + i

            params = {
                "page": current_page,
                "Size": PAGE_LIMIT,   # API expects capital S
                "order": order
            }

            resp = await client.get(
                NEPSELYTICS_FLOORSHEET_URL,
                params=params,
                headers=HEADERS
            )

            if resp.status_code != 200:
                raise HTTPException(
                    resp.status_code,
                    f"Floorsheet fetch failed at page {current_page}"
                )

            payload = resp.json()

            # Nepselytics response structure
            if not isinstance(payload, dict):
                break

            content = payload.get("data", {}).get("content", [])
            if not content:
                break

            all_records.extend(content)

            # Stop early if fewer than limit returned
            if len(content) < PAGE_LIMIT:
                break

    # Trim to requested size
    all_records = all_records[:size]

    return {
        "success": True,
        "count": len(all_records),
        "page": page,
        "requested_size": size,
        "data": all_records
    }
