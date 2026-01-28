from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import pandas as pd
import numpy as np
from io import StringIO
import json
from fastapi.responses import JSONResponse

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
# TECHNICAL DATA CONFIG (RSI & MA)
# -------------------------------------------------
GOOGLE_SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "1Q_En7VGGfifDmn5xuiF-t_02doPpwl4PLzxb4TBCW0Q"
    "/export?format=csv"
)

RSI_PERIOD = 14
MA_PERIOD = 20
MA_50 = 50
MA_200 = 200

# Initialize as empty DataFrames (Prevents 503 during loading)
RSI_LATEST_CACHE = pd.DataFrame()
MA_LATEST_CACHE = pd.DataFrame()
CROSSOVER_LATEST_CACHE = pd.DataFrame()
CONFLUENCE_LATEST_CACHE = pd.DataFrame()
CANDLESTICK_LATEST_CACHE = pd.DataFrame()
MOMENTUM_LATEST_CACHE = pd.DataFrame()
TECHNICAL_RAW_CACHE = None

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
            "/stock-chart/index/1D",
            "/rsi/all",
            "/rsi/filter?min=30&max=70",
            "/rsi/status",
            "/ma/all",
            "/ma/status",
            "/crossovers/all",
            "/confluence/all",
            "/candlesticks/all",
            "/momentum/all",
            "/refresh-technical"
        ]
    }

# -------------------------------------------------
# ShareHub Homepage Data
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
# TECHNICAL CALCULATIONS (ADVANCED)
# -------------------------------------------------
def calculate_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ma(close: pd.Series, period: int = 20):
    return close.rolling(window=period).mean()

def detect_candlestick(df):
    """Detects basic candlestick patterns on the last row"""
    if len(df) < 2: return "Neutral"
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    o, h, l, c = curr['open'], curr['high'], curr['low'], curr['close']
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return "Neutral"
    
    body_percent = body / candle_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # 1. HAMMER (Bullish)
    if lower_wick > (2 * body) and upper_wick < (0.1 * candle_range) and body_percent < 0.4:
        return "Hammer (Bullish)"
        
    # 2. SHOOTING STAR (Bearish)
    if upper_wick > (2 * body) and lower_wick < (0.1 * candle_range) and body_percent < 0.4:
        return "Shooting Star (Bearish)"
        
    # 3. BULLISH ENGULFING
    if c > o and prev['close'] < prev['open'] and c > prev['open'] and o < prev['close']:
        return "Bullish Engulfing"
        
    # 4. BEARISH ENGULFING
    if c < o and prev['close'] > prev['open'] and c < prev['open'] and o > prev['close']:
        return "Bearish Engulfing"
        
    return "Neutral"

def calculate_confluence_score(rsi, ma_dist_pct, sma50, sma200):
    """Calculates a Super-Signal score from 0-100"""
    score = 50 # Baseline
    
    # RSI Contribution (Oversold is good for scoring buy setups)
    if not pd.isna(rsi):
        if rsi < 30: score += 25
        elif rsi < 40: score += 15
        elif rsi > 70: score -= 20
        elif rsi > 60: score -= 10
        
    # MA distance (Price above MA is bullish)
    if not pd.isna(ma_dist_pct):
        if ma_dist_pct > 0: score += 10
        if ma_dist_pct > 5: score += 5
        
    # SMAs (SMA 50 > 200 is Golden)
    if not pd.isna(sma50) and not pd.isna(sma200):
        if sma50 > sma200: score += 15
        
    return max(0, min(100, score))

# -------------------------------------------------
# LOAD TECHNICAL DATA ON STARTUP
# -------------------------------------------------
@app.on_event("startup")
async def load_technical_data():
    global TECHNICAL_RAW_CACHE, RSI_LATEST_CACHE, MA_LATEST_CACHE, CROSSOVER_LATEST_CACHE, CONFLUENCE_LATEST_CACHE, CANDLESTICK_LATEST_CACHE, MOMENTUM_LATEST_CACHE

    try:
        print("🔄 Fetching Technical data from Google Sheets...")
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(GOOGLE_SHEET_CSV)

        if resp.status_code != 200:
            print(f"❌ Technical CSV fetch failed: {resp.status_code}")
            return

        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip().str.lower()
        
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print(f"❌ Missing required columns: {required - set(df.columns)}")
            return

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["date", "symbol", "close"])
        df = df.sort_values(["symbol", "date"])
        TECHNICAL_RAW_CACHE = df.copy()

        rsi_list, ma_list, cross_list, conf_list, candle_list, momentum_list = [], [], [], [], [], []

        for symbol, g in df.groupby("symbol"):
            symbol_str = str(symbol).upper()
            # Exclude Debentures and Mutual Funds (usually contain digits)
            if any(char.isdigit() for char in symbol_str):
                continue

            g = g.copy()
            data_len = len(g)
            if data_len < 2: continue
            
            # Indicators
            g["rsi"] = calculate_rsi(g["close"], RSI_PERIOD)
            g["ma20"] = calculate_ma(g["close"], MA_PERIOD)
            g["sma50"] = calculate_ma(g["close"], MA_50)
            g["sma200"] = calculate_ma(g["close"], MA_200)
            g["vol_avg20"] = calculate_ma(g["volume"], 20)
            
            last = g.iloc[-1]
            prev = g.iloc[-2]
            
            # RSI Results
            if not pd.isna(last["rsi"]):
                rsi_list.append({"symbol": str(symbol).upper(), "close": float(last["close"]), "rsi": round(float(last["rsi"]), 2)})

            # MA 20 Results
            ma_dist_pct = 0
            if not pd.isna(last["ma20"]):
                ma_dist_pct = (last["close"] - last["ma20"]) / last["ma20"] * 100
                ma_list.append({
                    "symbol": str(symbol).upper(), "close": float(last["close"]),
                    "ma": round(float(last["ma20"]), 2), "percent_diff": round(float(ma_dist_pct), 2)
                })

            # Crossover Results
            if data_len >= MA_200 and not pd.isna(last["sma200"]):
                signal = "Golden Cross" if (prev["sma50"] <= prev["sma200"] and last["sma50"] > last["sma200"]) else \
                         "Death Cross" if (prev["sma50"] >= prev["sma200"] and last["sma50"] < last["sma200"]) else \
                         "Bullish Alignment" if last["sma50"] > last["sma200"] else "Bearish Alignment"
                
                cross_list.append({
                    "symbol": str(symbol).upper(), "close": float(last["close"]),
                    "sma50": round(float(last["sma50"]), 2), "sma200": round(float(last["sma200"]), 2),
                    "signal": signal, "is_cross": ("Cross" in signal and "Alignment" not in signal)
                })

            # Candlestick Results
            pattern = detect_candlestick(g)
            if pattern != "Neutral":
                candle_list.append({"symbol": str(symbol).upper(), "close": float(last["close"]), "pattern": pattern})

            # Confluence Result
            score = calculate_confluence_score(last["rsi"], ma_dist_pct, last["sma50"], last["sma200"])
            conf_list.append({
                "symbol": str(symbol).upper(), "close": float(last["close"]), "score": int(score),
                "rsi": round(float(last["rsi"]), 2) if not pd.isna(last["rsi"]) else None,
                "trend": "Bullish" if score > 60 else "Bearish" if score < 40 else "Neutral"
            })

            # Momentum IQ (Volume Shocker, 52-Week High, RS)
            vol_ratio = 0
            v_avg = last["vol_avg20"] if not pd.isna(last["vol_avg20"]) else 0
            if v_avg > 0:
                vol_ratio = last["volume"] / v_avg

            # 52-Week logic (approx 250 days)
            win_52 = g.tail(250)
            h52 = win_52["high"].max()
            l52 = win_52["low"].min()
            
            high_52 = float(h52) if not pd.isna(h52) else float(last["close"])
            low_52 = float(l52) if not pd.isna(l52) else float(last["close"])
            
            within_high = False
            within_low = False
            if high_52 > 0:
                within_high = (high_52 - last["close"]) / high_52 <= 0.02
            if low_52 > 0:
                within_low = (last["close"] - low_52) / low_52 <= 0.02
            
            # Simple RS Score: (Current Price / Price 1 Year Ago or 250 days ago)
            rs_score = 0
            if data_len >= 250:
                start_price = g.iloc[-250]["close"]
                if not pd.isna(start_price) and start_price > 0:
                    rs_score = (last["close"] / start_price) * 100

            momentum_list.append({
                "symbol": str(symbol).upper(),
                "close": float(last["close"]),
                "volume": int(last["volume"]) if not pd.isna(last["volume"]) else 0,
                "vol_avg20": round(float(v_avg), 0),
                "vol_ratio": round(float(vol_ratio), 2),
                "high_52": high_52,
                "low_52": low_52,
                "rs_score": round(float(rs_score), 2),
                "breakout": "High" if last["close"] >= high_52 else "Low" if last["close"] <= low_52 else "Near High" if within_high else "Near Low" if within_low else "Neutral"
            })

        RSI_LATEST_CACHE = pd.DataFrame(rsi_list)
        MA_LATEST_CACHE = pd.DataFrame(ma_list)
        CROSSOVER_LATEST_CACHE = pd.DataFrame(cross_list)
        CANDLESTICK_LATEST_CACHE = pd.DataFrame(candle_list)
        CONFLUENCE_LATEST_CACHE = pd.DataFrame(conf_list)
        MOMENTUM_LATEST_CACHE = pd.DataFrame(momentum_list)

        print(f"✅ Technical Data Loaded: RSI:{len(rsi_list)}, MA:{len(ma_list)}, Conf:{len(conf_list)}, Mom:{len(momentum_list)}")

    except Exception as e:
        print("❌ Startup Load Error:", str(e))
        import traceback; traceback.print_exc()
        # Ensure caches are empty if an error occurs during processing
        TECHNICAL_RAW_CACHE = pd.DataFrame()
        RSI_LATEST_CACHE = pd.DataFrame()
        MA_LATEST_CACHE = pd.DataFrame()
        CROSSOVER_LATEST_CACHE = pd.DataFrame()
        CONFLUENCE_LATEST_CACHE = pd.DataFrame()
        CANDLESTICK_LATEST_CACHE = pd.DataFrame()
        MOMENTUM_LATEST_CACHE = pd.DataFrame()

# -------------------------------------------------
# TECHNICAL ENDPOINTS (RSI & MA)
# -------------------------------------------------
@app.get("/rsi/all")
def rsi_all():
    return JSONResponse(content=json.loads(RSI_LATEST_CACHE.to_json(orient="records")))

@app.get("/ma/all")
def ma_all():
    return JSONResponse(content=json.loads(MA_LATEST_CACHE.to_json(orient="records")))

@app.get("/ma/status")
def ma_status():
    return {
        "status": "ready" if not MA_LATEST_CACHE.empty else "not_ready",
        "symbols": len(MA_LATEST_CACHE),
        "period": MA_PERIOD
    }

@app.get("/crossovers/all")
def crossovers_all():
    return JSONResponse(content=json.loads(CROSSOVER_LATEST_CACHE.to_json(orient="records")))

@app.get("/confluence/all")
def confluence_all():
    if CONFLUENCE_LATEST_CACHE.empty: return []
    df = CONFLUENCE_LATEST_CACHE.sort_values("score", ascending=False)
    return JSONResponse(content=json.loads(df.to_json(orient="records")))

@app.get("/candlesticks/all")
def candlesticks_all():
    return JSONResponse(content=json.loads(CANDLESTICK_LATEST_CACHE.to_json(orient="records")))

@app.get("/momentum/all")
def momentum_all():
    return JSONResponse(content=json.loads(MOMENTUM_LATEST_CACHE.to_json(orient="records")))

@app.get("/refresh-technical")
async def refresh_technical():
    """Trigger a manual refresh of all technical data from Google Sheets"""
    await load_technical_data()
    return {"status": "success", "message": "Technical data refreshed."}

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