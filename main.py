from fastapi import FastAPI, HTTPException, Query
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
NEPSELYTICS_FLOORSHEET_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/floorsheet"

# Root endpoint
@app.get("/")
def root():
    return {"status": "NEPSE Data API Running", "endpoints": ["/homepage-data", "/market-turnover", "/floorsheet"]}

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

# -------------------------------
# Floorsheet Data (Live Trading)
# -------------------------------
@app.get("/floorsheet")
async def floorsheet(
    page: int = Query(0, ge=0, description="Page number (0-indexed)"),
    size: int = Query(500, ge=1, le=500, description="Number of records per page (max 500)"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
):
    """
    Fetch live floorsheet data from NEPSE.
    
    - **page**: Page number (starts from 0)
    - **size**: Records per page (1-500, API may limit to 100 per request)
    - **order**: Sort order ('asc' or 'desc')
    
    Returns combined data if size > 100 (API limitation workaround)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    # If requested size is > 100, fetch multiple pages and combine
    if size > 100:
        all_records = []
        pages_needed = (size + 99) // 100  # Ceiling division
        
        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(pages_needed):
                current_page = page + i
                params = {
                    "page": current_page,
                    "Size": 100,  # API uses capital 'S'
                    "order": order
                }
                
                resp = await client.get(NEPSELYTICS_FLOORSHEET_URL, params=params, headers=headers)
                
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Failed to fetch floorsheet data (page {current_page})"
                    )
                
                data = resp.json()
                
                # Extract records from nepselytics format
                if isinstance(data, dict) and 'data' in data:
                    records = data['data'].get('content', [])
                    all_records.extend(records)
                    
                    # Stop if we got fewer records than requested (last page)
                    if len(records) < 100:
                        break
                else:
                    break
        
        # Trim to requested size
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
        # Single page request
        params = {
            "page": page,
            "Size": size,
            "order": order
        }
        
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(NEPSELYTICS_FLOORSHEET_URL, params=params, headers=headers)
        
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail="Failed to fetch floorsheet data"
            )
        
        return resp.json()
