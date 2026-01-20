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
    page: int = Query(0, ge=0),
    size: int = Query(10, ge=1, le=500),
    order: str = Query("desc", regex="^(asc|desc)$")
):
    PAGE_LIMIT = 100
    pages_required = (size + PAGE_LIMIT - 1) // PAGE_LIMIT

    all_content = []
    base_meta = None

    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(pages_required):
            current_page = page + i

            params = {
                "page": current_page,
                "Size": min(PAGE_LIMIT, size),
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
                    f"Failed to fetch floorsheet page {current_page}"
                )

            payload = resp.json()
            data = payload.get("data")

            if not data or not data.get("content"):
                break

            # Store metadata from first page only
            if base_meta is None:
                base_meta = data

            all_content.extend(data["content"])

            if len(data["content"]) < PAGE_LIMIT:
                break

    # Trim to requested size
    all_content = all_content[:size]

    # Build Nepselytics-compatible response
    return {
        "success": True,
        "code": None,
        "message": None,
        "data": {
            "totalAmount": base_meta.get("totalAmount", 0),
            "totalQty": base_meta.get("totalQty", 0),
            "totalTrades": base_meta.get("totalTrades", 0),
            "pageIndex": page,
            "totalPages": base_meta.get("totalPages", 0),
            "totalItems": base_meta.get("totalItems", 0),
            "pageSize": size,
            "content": all_content
        }
    }

