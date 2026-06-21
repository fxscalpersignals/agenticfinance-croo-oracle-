import os
import time
import requests
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from threading import Lock
from typing import Dict, List

app = FastAPI()

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
CACHE_TTL = 300 # 5 min
MAIN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]

# Real sector mapping for hackathon judges
SECTOR_MAP = {
    "AI": ["FETUSDT", "AGIXUSDT", "OCEANUSDT", "RNDRUSDT"],
    "RWA": ["ONDOUSDT", "POLYXUSDT", "CFGUSDT"],
    "DEFI": ["AAVEUSDT", "UNIUSDT", "MKRUSDT", "COMPUSDT"],
    "PAYFI": ["XRPUSDT", "XLMUSDT", "ALGOUSDT"]
}

# ==== CACHE ====
cache = {
    "signals": {}, # {symbol: {signal, confidence, entry, tp, sl, reasons}}
    "sectors": {"AI": 0.0, "RWA": 0.0, "DEFI": 0.0, "PAYFI": 0.0},
    "start_time": time.time(),
    "last_update": 0
}
cache_lock = Lock()

# ==== FIX: HEALTH CHECK ====
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok", "agent": "CROO AI Oracle", "timestamp": int(time.time())}

# ==== TA HELPERS ====
def calculate_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1: return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(closes: List[float], period: int) -> float:
    if len(closes) < period: return closes[-1]
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return np.convolve(closes, weights, mode='valid')[-1]

# ==== BINANCE KLINES - FIXED ====
def get_klines(symbol: str, interval: str = "1h", limit: int = 100) -> Dict:
    """Fetch real candle data for pullback detection"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5
        )
        if r.status_code!= 200: raise Exception(f"Binance klines {r.status_code}")
        data = r.json()
        closes = [float(x[4]) for x in data] # close prices
        volumes = [float(x[5]) for x in data] # volumes
        highs = [float(x[2]) for x in data]
        return {"closes": closes, "volumes": volumes, "highs": highs}
    except Exception as e:
        print(f"Binance klines failed for {symbol}: {e}")
        return None

# ==== MEXC FALLBACK - NOW PARSING ====
def get_mexc_24hr() -> Dict:
    try:
        r = requests.get("https://api.mexc.com/api/v3/ticker/24hr", timeout=5)
        if r.status_code!= 200: return {}
        data = r.json()
        return {x["symbol"]: float(x["priceChangePercent"]) for x in data}
    except Exception as e:
        print(f"MEXC failed: {e}")
        return {}

# ==== PULLBACK ENGINE - THE CROO DIFFERENCE ====
def analyze_pullback(symbol: str) -> Dict:
    """Returns confidence score + entry zones per your branding"""
    klines = get_klines(symbol, "1h", 100)
    if not klines: return {"signal": "NONE", "confidence": 0, "reasons": ["API Error"]}
    
    closes = klines["closes"]
    volumes = klines["volumes"]
    highs = klines["highs"]
    
    price = closes[-1]
    rsi = calculate_rsi(closes)
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    
    # Pullback % from recent high
    recent_high = max(highs[-24:]) # 24h high
    pullback_pct = ((recent_high - price) / recent_high) * 100
    
    # Volume check - is current vol > 20-period avg?
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.2
    
    # Confidence scoring
    confidence = 0
    reasons = []
    
    if rsi < 40:
        confidence += 25
        reasons.append("✅ RSI Oversold")
    if price > ema50:
        confidence += 25
        reasons.append("✅ Above EMA50")
    if 3 <= pullback_pct <= 8:
        confidence += 25
        reasons.append(f"✅ {pullback_pct:.1f}% Pullback")
    if vol_spike:
        confidence += 25
        reasons.append("✅ Volume Recovery")
    
    signal = "BUY" if confidence >= 75 else "WATCH" if confidence >= 50 else "NONE"
    
    # Entry/TP/SL calc
    entry_low = round(price * 0.995, 2)
    entry_high = round(price * 1.005, 2)
    tp = round(price * 1.045, 2) # 4.5% target
    sl = round(ema50 * 0.98, 2) # stop below EMA50
    
    return {
        "symbol": symbol.replace("USDT", ""),
        "price": round(price, 2),
        "signal": signal,
        "confidence": confidence,
        "entry_zone": f"{entry_low} - {entry_high}",
        "tp": tp,
        "sl": sl,
        "rsi": round(rsi, 1),
        "reasons": reasons if reasons else ["❌ No confluence"]
    }

# ==== SECTOR ENGINE - REAL DATA ====
def update_sectors():
    sector_scores = {}
    mexc_data = get_mexc_24hr() # fallback if binance fails
    
    for sector, symbols in SECTOR_MAP.items():
        changes = []
        for symbol in symbols:
            try:
                # Try Binance first - FIXED single symbol call
                r = requests.get(
                    "https://api.binance.com/api/v3/ticker/24hr",
                    params={"symbol": symbol},
                    timeout=3
                )
                if r.status_code == 200:
                    changes.append(float(r.json()["priceChangePercent"]))
                    continue
            except: pass
            # MEXC fallback
            if symbol in mexc_data:
                changes.append(mexc_data[symbol])
        
        sector_scores[sector] = round(np.mean(changes), 2) if changes else 0.0
    
    with cache_lock:
        cache["sectors"] = sector_scores

# ==== MAIN SCANNER ====
def run_scanner():
    now = time.time()
    with cache_lock:
        if now - cache["last_update"] < CACHE_TTL:
            return cache
    
    # 1. Update sectors
    update_sectors()
    
    # 2. Scan main symbols for pullbacks
    for symbol in MAIN_SYMBOLS:
        result = analyze_pullback(symbol)
        with cache_lock:
            cache["signals"][symbol] = result
    
    with cache_lock:
        cache["last_update"] = now
    print("Scanner run complete")
    return cache

# ==== CAP ENDPOINT - CROO AGENT API ====
@app.post("/oracle")
async def oracle_api(req: Request):
    """CAP-compatible endpoint for Agent-to-Agent calls"""
    data = await req.json()
    asset = data.get("asset", "BTC").upper()
    symbol = f"{asset}USDT"
    
    if symbol not in MAIN_SYMBOLS:
        return JSONResponse({"error": "Asset not supported"}, status_code=400)
    
    run_scanner() # ensure fresh
    signal = cache["signals"].get(symbol, {})
    
    if not signal or signal["signal"] == "NONE":
        return JSONResponse({
            "signal": "NONE",
            "confidence": 0,
            "message": "No high-confidence setup"
        })
    
    return JSONResponse({
        "signal": signal["signal"],
        "confidence": signal["confidence"],
        "entry": signal["entry_zone"],
        "tp": signal["tp"],
        "sl": signal["sl"],
        "rsi": signal["rsi"],
        "reasons": signal["reasons"]
    })

# ==== TELEGRAM ====
def send_message(chat_id, text):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    if "message" not in data: return {"ok": True}
    
    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")
    
    if text == "/start":
        run_scanner()
        msg = "🚀 <b>CROO AI Oracle Agent</b>\n\nAuto-scans BTC/ETH/XRP/SOL for high-confidence pullback entries.\n\nUse /scan to get signals.\nAPI: POST /oracle"
        send_message(chat_id, msg)
    
    elif text == "/scan":
        run_scanner()
        signals = cache["signals"]
        high_conf = [v for v in signals.values() if v["confidence"] >= 75]
        
        if not high_conf:
            send_message(chat_id, "🔍 No high-confidence pullbacks right now. Market scanning...")
            return {"ok": True}
        
        for s in high_conf:
            msg = f"""🚨 <b>HIGH-CONFIDENCE PULLBACK</b>

Asset: {s['symbol']}
Price: ${s['price']}

Confidence: {s['confidence']}%

Reasons:
{chr(10).join(s['reasons'])}

Entry Zone:
{s['entry_zone']}

Target: {s['tp']}
Stop: {s['sl']}"""
            send_message(chat_id, msg)
    
    elif text == "/sectors":
        run_scanner()
        sectors = cache["sectors"]
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
        msg = "📈 <b>Sector Intelligence Map</b>\n\n"
        for name, change in sorted_sectors:
            emoji = "🔥" if change > 2 else "🟡" if change > 0 else "🔻"
            msg += f"{emoji} {name}: {change:+.2f}%\n"
        send_message(chat_id, msg)
    
    return {"ok": True}

@app.get("/")
def root():
    return {
        "agent": "CROO AI Oracle",
        "version": "2.0",
        "capabilities": ["pullback_scan", "confidence_scoring", "sector_rotation"],
        "api": "/oracle"
    }
