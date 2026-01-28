from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import pandas as pd
import numpy as np
from io import StringIO

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
NEPSELYTICS_URL = "https://sharehubnepal.com/live/api/v2/nepselive/home-page-data"
NEPSE_TURNOVER_URL = "https://tms59.nepsetms.com.np/tmsapi/rtApi/admin/vCache/marketTurnover"
NEPSELYTICS_FLOORSHEET_URL = "https://nepselytics-6d61dea19f30.herokuapp.com/api/nepselytics/floorsheet"

NEPALIPAISA_INDEX_URL = "https://nepalipaisa.com/api/GetIndexLive"
NEPALIPAISA_SUBINDEX_URL = "https://nepalipaisa.com/api/GetSubIndexLive"

SHAREHUB_ANNOUNCEMENT_URL = "https://sharehubnepal.com/data/api/v1/announcement"

# -------------------------------------------------
# RSI – Google Sheet Config
# -------------------------------------------------
GOOGLE_SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "1Q_En7VGGfifDmn5xuiF-t_02doPpwl4PLzxb4TBCW0Q"
    "/export?format=csv"
)

RSI_PERIOD = 14
RSI_RAW_CACHE = None
RSI_LATEST_CACHE = None

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
            "/subindex-live",
            "/floorsheet",
            "/floorsheet/totals",
            "/announcements",
            "/stock-chart/{symbol}?time=1D|1W|1M|3M|6M|1Y",
            "/rsi/all",
            "/rsi/symbol",
            "/rsi/filter",
            "/stock-chart/index/1D",
            "/stock-chart/{symbol}"
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
# NEPSE Sub-Index Live (NepaliPaisa)
# -------------------------------------------------
@app.get("/subindex-live")
async def subindex_live():
    """
    Live NEPSE sub-index data (Banking, Hydro, Dev Bank, Finance, etc.)
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
            NEPALIPAISA_SUBINDEX_URL,
            headers=headers,
            params=params
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch NEPSE sub-index live data"
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
        pages_needed = (size + 99) // 100

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

        return {
            "success": True,
            "message": f"Fetched {len(all_records[:size])} records",
            "data": all_records[:size],
            "pagination": {
                "page": page,
                "size": len(all_records[:size]),
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

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# -------------------------------------------------
# ShareHub Nepal Announcements
# -------------------------------------------------
@app.get("/announcements")
async def announcements(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(12, ge=1, le=50, description="Items per page")
):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    params = {
        "Page": page,
        "Size": size
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            SHAREHUB_ANNOUNCEMENT_URL,
            params=params,
            headers=headers
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch ShareHub announcements"
        )

    return resp.json()

# -------------------------------------------------
# Stock Chart Data (ShareHub Nepal Price History)
# -------------------------------------------------
@app.get("/stock-chart/{symbol}")
async def stock_chart(
    symbol: str,
    time: str = Query(
        "1Y",
        regex="^(1D|1W|1M|3M|6M|1Y|5Y)$",
        description="Timeframe: 1D, 1W, 1M, 3M, 6M, 1Y, 5Y"
    )
):
    """
    Proxy endpoint for ShareHub Nepal price history graph API
    """

    # URL construction
    base_url = "https://sharehubnepal.com/data/api/v1/price-history/graph"
    url = f"{base_url}/{symbol.upper()}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    params = {
        "time": time
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to fetch price history for {symbol.upper()}"
        )

    return resp.json()


@app.get("/stock-chart/index/1D")
async def index_1d_chart():
    """
    Fetch 1D intraday index chart using ShareHub's daily-graph API.
    """
    url = "https://sharehubnepal.com/live/api/v1/daily-graph/index/58"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail="Failed to fetch 1D index graph"
        )

    return resp.json()

# -------------------------------------------------
# RSI CALCULATION
# -------------------------------------------------
def calculate_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# -------------------------------------------------
# LOAD RSI ON STARTUP (RAILWAY SAFE)
# -------------------------------------------------
@app.on_event("startup")
async def load_rsi_data():
    global RSI_RAW_CACHE, RSI_LATEST_CACHE

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(GOOGLE_SHEET_CSV)

        if resp.status_code != 200:
            print(f"❌ RSI CSV fetch failed: {resp.status_code}")
            RSI_LATEST_CACHE = pd.DataFrame()
            return

        df = pd.read_csv(StringIO(resp.text))

        # Normalize column names (CRITICAL)
        df.columns = df.columns.str.strip().str.lower()

        required = {"date", "symbol", "close"}
        if not required.issubset(df.columns):
            print("❌ Missing required columns:", df.columns.tolist())
            RSI_LATEST_CACHE = pd.DataFrame()
            return

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna(subset=["date", "symbol", "close"])
        df = df.sort_values(["symbol", "date"])

        results = []

        for symbol, g in df.groupby("symbol"):
            if len(g) < RSI_PERIOD + 1:
                continue

            g["rsi"] = calculate_rsi(g["close"], RSI_PERIOD)
            last = g.iloc[-1]

            if pd.isna(last["rsi"]):
                continue

            results.append({
                "symbol": symbol.upper(),
                "close": float(last["close"]),
                "rsi": round(float(last["rsi"]), 2)
            })

        RSI_LATEST_CACHE = pd.DataFrame(results)

        print(f"✅ RSI Loaded for {len(RSI_LATEST_CACHE)} symbols")

    except Exception as e:
        print("❌ RSI startup exception:", str(e))
        RSI_LATEST_CACHE = pd.DataFrame()

# -------------------------------------------------
# RSI ENDPOINTS
# -------------------------------------------------
@app.get("/rsi/all")
def rsi_all():
    if RSI_LATEST_CACHE is None or RSI_LATEST_CACHE.empty:
        raise HTTPException(status_code=503, detail="RSI data not ready")
    return RSI_LATEST_CACHE.to_dict(orient="records")

@app.get("/rsi/symbol")
def rsi_symbol(symbol: str = Query(...)):
    if RSI_LATEST_CACHE is None:
        raise HTTPException(status_code=503, detail="RSI data not ready")

    symbol = symbol.upper()
    row = RSI_LATEST_CACHE[RSI_LATEST_CACHE["symbol"] == symbol]

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    return row.iloc[0].to_dict()

@app.get("/rsi/filter")
def rsi_filter(
    min: float | None = Query(None),
    max: float | None = Query(None)
):
    if RSI_LATEST_CACHE is None:
        raise HTTPException(status_code=503, detail="RSI data not ready")

    df = RSI_LATEST_CACHE.copy()

    if min is not None:
        df = df[df["rsi"] >= min]
    if max is not None:
        df = df[df["rsi"] <= max]

    return df.sort_values("rsi").to_dict(orient="records")

@app.get("/rsi/status")
def rsi_status():
    if RSI_LATEST_CACHE is None:
        return {"status": "not_loaded"}
    if RSI_LATEST_CACHE.empty:
        return {"status": "loaded_but_empty"}
    return {
        "status": "ready",
        "symbols": len(RSI_LATEST_CACHE)
    }

# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "NEPSE API running",
        "rsi_symbols": 0 if RSI_LATEST_CACHE is None else len(RSI_LATEST_CACHE)
    }