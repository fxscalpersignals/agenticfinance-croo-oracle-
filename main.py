import os
import time
import requests
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

app = FastAPI()

# ===== CONFIG =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
CHAT_ID = os.environ.get("CHAT_ID")

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

# 9 coins for hackathon demo
ASSETS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT",
    "DOGEUSDT","ADAUSDT","TRXUSDT","LINKUSDT","AVAXUSDT"
]

# ===== STATE =====
cache = {"signals": {}, "last_scan": 0}
signal_history = []
last_alerted = {}

# ===== HELPERS =====
def get_binance_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=5)
        return r.json()
    except:
        return None

def get_mexc_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.mexc.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=5)
        return r.json()
    except:
        return None

def get_coingecko_price(asset):
    mapping = {
        "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "XRPUSDT": "ripple",
        "SOLUSDT": "solana", "DOGEUSDT": "dogecoin", "ADAUSDT": "cardano",
        "TRXUSDT": "tron", "LINKUSDT": "chainlink", "AVAXUSDT": "avalanche-2"
    }
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": mapping[asset], "vs_currencies": "usd"},
            timeout=5
        )
        return float(r.json()[mapping[asset]]["usd"])
    except:
        return None

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.array([50.0] * len(closes))
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down!= 0 else 0
    rsi = np.zeros_like(closes)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(closes)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down!= 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calc_ema(closes, period):
    if len(closes) < period:
        return np.array([closes[-1]] if len(closes) > 0 else [0])
    return np.convolve(closes, np.ones(period)/period, mode='valid')

def analyze_asset(symbol, timeframe="1h"):
    klines = get_binance_klines(symbol, timeframe)
    if not klines:
        klines = get_mexc_klines(symbol, timeframe)
    
    if not klines or len(klines) < 50:
        price = get_coingecko_price(symbol)
        if not price:
            return None
        return {"asset": symbol, "price": price, "confidence": 0, "signal": "NONE", "rsi": 50, "reasons": [], "ema20": price, "ema50": price, "pullback_pct": 0}
    
    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    
    rsi = calc_rsi(closes)[-1]
    ema20_arr = calc_ema(closes, 20)
    ema50_arr = calc_ema(closes, 50)
    ema20 = ema20_arr[-1] if len(ema20_arr) > 0 else price
    ema50 = ema50_arr[-1] if len(ema50_arr) > 0 else price
    
    recent_high = max(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5
    
    confidence = 0
    reasons = []
    
    if rsi < 40:
        confidence += 25
        reasons.append("RSI Oversold")
    if price > ema50:
        confidence += 25
        reasons.append("Above EMA50")
    if 3 < pullback < 10:
        confidence += 25
        reasons.append("Healthy Pullback")
    if vol_spike:
        confidence += 25
        reasons.append("Volume Spike")
    
    if confidence >= 60:
        signal = "BUY"
    elif confidence >= 35:
        signal = "WATCH"
    else:
        signal = "NONE"
    
    return {
        "asset": symbol,
        "price": round(price, 4),
        "rsi": round(rsi, 1),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "pullback_pct": round(pullback, 2),
        "confidence": confidence,
        "signal": signal,
        "reasons": reasons,
        "timestamp": datetime.utcnow().isoformat()
    }

def run_scanner():
    global signal_history
    
    results = {}
    for asset in ASSETS:
        data = analyze_asset(asset, "1h")
        if data:
            results[asset] = data
            
            if data["confidence"] >= 75 and asset not in last_alerted and bot:
                asyncio.create_task(send_auto_alert(asset, data))
                last_alerted[asset] = time.time()
            
            if data["signal"] in ["BUY", "WATCH"]:
                signal_history.append(data)
    
    signal_history = signal_history[-50:]
    cache["signals"] = results
    cache["last_scan"] = time.time()
    return results

async def send_auto_alert(asset, data):
    if not CHAT_ID or not bot:
        return
    msg = f"🚨 AUTO ALERT: {asset}\nConfidence: {data['confidence']}/100\nPrice: ${data['price']}\nReasons: {', '.join(data['reasons'])}"
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        print(f"Alert failed: {e}")

@app.on_event("startup")
async def startup_event():
    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
        print(f"Webhook set: {webhook_url}")
    run_scanner()

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        await handle_message(chat_id, text)
    elif "callback_query" in data:
        query = data["callback_query"]
        chat_id = query["message"]["chat"]["id"]
        data_btn = query["data"]
        await handle_callback(chat_id, data_btn)
    return JSONResponse({"ok": True})

async def handle_message(chat_id, text):
    if text == "/start" and bot:
        keyboard = [
            [InlineKeyboardButton("📊 Scan All", callback_data="scan_all"),
             InlineKeyboardButton("📈 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("🔍 BTC", callback_data="BTCUSDT"),
             InlineKeyboardButton("🔍 ETH", callback_data="ETHUSDT")],
            [InlineKeyboardButton("🔍 XRP", callback_data="XRPUSDT"),
             InlineKeyboardButton("🔍 SOL", callback_data="SOLUSDT")],
            [InlineKeyboardButton("📊 Status", callback_data="status")]
        ]
        msg = "📊 Pullback Scan Results\n\nNo high-confidence pullbacks right now.\nMarket scanning continues 24/7."
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_callback(chat_id, data):
    if not bot:
        return
    if data == "scan_all":
        run_scanner()
        signals = cache["signals"]
        buys = [s for s in signals.values() if s["signal"] == "BUY"]
        watch = [s for s in signals.values() if s["signal"] == "WATCH"]
        
        if buys:
            msg = "🚨 BUY SIGNALS:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: {s['confidence']}/100" for s in buys])
        elif watch:
            msg = "👀 WATCH LIST:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: {s['confidence']}/100" for s in watch])
        else:
            msg = "No high-confidence pullbacks right now."
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data in ASSETS:
        run_scanner()
        s = cache["signals"].get(data)
        if not s or s["signal"] == "NONE":
            msg = f"No setup for {data.replace('USDT','')} right now.\nRSI: {s['rsi'] if s else 'N/A'}\nUse 📊 Scan All to refresh."
        else:
            msg = f"{s['signal']}: {s['asset'].replace('USDT','')}\nPrice: ${s['price']}\nConfidence: {s['confidence']}/100\nRSI: {s['rsi']}\nReasons: {', '.join(s['reasons'])}"
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data == "status":
        uptime = int(time.time() - start_time)
        last = int(time.time() - cache["last_scan"]) if cache["last_scan"] else 0
        msg = f"📊 CROO Oracle Status\nUptime: {uptime//3600}h {(uptime%3600)//60}m\nLast scan: {last}s ago\nSignals tracked: {len(signal_history)}"
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data == "leaderboard":
        buys = [s for s in signal_history if s["signal"] == "BUY"][-5:]
        if not buys:
            msg = "No BUY signals yet."
        else:
            msg = "🏆 Recent BUY Signals:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: {s['confidence']}/100 @ ${s['price']}" for s in buys])
        await bot.send_message(chat_id=chat_id, text=msg)

@app.get("/")
def root():
    return {"status": "CROO Oracle Online", "uptime": int(time.time() - start_time)}

@app.get("/debug")
def debug():
    run_scanner()
    return cache["signals"]

@app.get("/oracle")
def oracle(asset: str = None):
    run_scanner()
    if asset:
        return cache["signals"].get(asset.upper(), {"error": "Asset not found"})
    return cache["signals"]

@app.post("/oracle")
async def oracle_post(request: Request):
    data = await request.json()
    asset = data.get("asset", "").upper()
    run_scanner()
    return cache["signals"].get(asset, {"error": "Asset not found"})

@app.get("/stats")
def stats():
    total = len(signal_history)
    wins = sum(1 for s in signal_history if s.get("pnl", 0) > 0)
    return {
        "total_signals": total,
        "win_rate": round(wins/total*100, 1) if total else 0,
        "last_signal": signal_history[-1] if signal_history else None
    }

@app.get("/cap/health")
def cap_health():
    return {
        "agent": "CROO Oracle",
        "status": "active",
        "assets": ASSETS,
        "uptime_seconds": int(time.time() - start_time)
    }
