import os
import time
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from threading import Lock

app = FastAPI()

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
CACHE_TTL = 300 # 5 min cache so free APIs don't rate limit you

# ==== CACHE ====
cache = {
    "ssi": {
        "score": 23, 
        "mood": "Extreme Fear", 
        "avg_change": 0.0, 
        "strongest": "N/A", 
        "strongest_change": -100.0
    },
    "sectors": {"AI": 0.0, "PAYFI": 0.0, "RWA": 0.0, "DEFI": 0.0},
    "start_time": time.time(),
    "last_update": 0,
    "total_signals": 0
}
cache_lock = Lock()

# ==== FIX: HEALTH CHECK - STOPS 405 ERRORS ====
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}

# ==== API FETCH WITH FALLBACK ====
def fetch_market_data():
    now = time.time()
    
    # Return cache if fresh
    with cache_lock:
        if now - cache["last_update"] < CACHE_TTL:
            return cache

    # Try Binance first
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbols": '["BTCUSDT","ETHUSDT","XRPUSDT","SOLUSDT"]'},
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            # Parse coins
            coins = {i['symbol']: float(i['priceChangePercent']) for i in data}
            avg = sum(coins.values()) / len(coins) if coins else 0
            strongest = max(coins, key=coins.get) if coins else "N/A"
            
            with cache_lock:
                cache["ssi"]["avg_change"] = round(avg, 2)
                cache["ssi"]["score"] = max(0, min(100, int(50 + avg * 2))) # basic SSI calc
                cache["ssi"]["mood"] = "Extreme Fear" if avg < -5 else "Fear" if avg < -2 else "Neutral" if avg < 2 else "Greed"
                cache["ssi"]["strongest"] = strongest.replace("USDT", "")
                cache["ssi"]["strongest_change"] = round(coins.get(strongest, 0), 2)
                cache["last_update"] = now
            print("Binance success")
            return cache
    except Exception as e:
        print(f"Binance failed: {e}")

    # Fallback to MEXC
    try:
        r = requests.get("https://api.mexc.com/api/v3/ticker/24hr", timeout=5)
        if r.status_code == 200:
            # Add MEXC parsing if you want
            with cache_lock:
                cache["last_update"] = now
            print("MEXC success")
            return cache
    except Exception as e:
        print(f"MEXC failed: {e}")

    # Both failed - return stale cache instead of zeros
    print("All APIs failed. Returning cached data.")
    return cache

# ==== TELEGRAM HELPERS ====
def send_message(chat_id, text, reply_markup=None):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram send error: {e}")

def get_main_keyboard():
    return {
        "keyboard": [
            [{"text": "📊 BTC"}, {"text": "📊 ETH"}],
            [{"text": "📊 XRP"}, {"text": "📊 SOL"}],
            [{"text": "📈 Sector Map"}, {"text": "🐋 Whale Radar"}],
            [{"text": "📊 ETF Flows"}, {"text": "🧠 Intelligence"}],
            [{"text": "📈 Performance"}, {"text": "📊 Stats"}],
            [{"text": "🔄 Scanner Status"}, {"text": "💰 Trade Now"}],
            [{"text": "💼 Portfolio"}]
        ],
        "resize_keyboard": True
    }

def format_uptime():
    seconds = int(time.time() - cache["start_time"])
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"

# ==== TELEGRAM WEBHOOK ====
@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    
    if "message" not in data:
        return {"ok": True}
    
    message = data["message"]
    chat_id = message["chat"]["id"]
    text = message.get("text", "")

    if text == "/start":
        market = fetch_market_data()
        ssi = market["ssi"]
        msg = f"""🧠 <b>Market Intelligence Feed</b>

SSI Score: {ssi['score']}/100
Mood: {ssi['mood']}
Avg Change: {ssi['avg_change']:+.2f}%
Strongest: {ssi['strongest']} ({ssi['strongest_change']:+.2f}%)

💡 SSI = Smart Sentiment Index"""
        send_message(chat_id, msg, get_main_keyboard())
        
    elif text == "🧠 Intelligence":
        market = fetch_market_data()
        ssi = market["ssi"]
        msg = f"""🧠 <b>Market Intelligence Feed</b>

SSI Score: {ssi['score']}/100
Mood: {ssi['mood']}
Avg Change: {ssi['avg_change']:+.2f}%
Strongest: {ssi['strongest']} ({ssi['strongest_change']:+.2f}%)

💡 SSI = Smart Sentiment Index"""
        send_message(chat_id, msg)
        
    elif text == "📈 Sector Map":
        market = fetch_market_data()
        sectors = market["sectors"]
        msg = f"""📈 <b>Sector Intelligence Map</b>

🔥 AI: {sectors['AI']:+.2f}% (Strongest)
🟡 PAYFI: {sectors['PAYFI']:+.2f}%
🟡 RWA: {sectors['RWA']:+.2f}%
🟡 DEFI: {sectors['DEFI']:+.2f}%

💡 Strongest sector = highest rotation"""
        send_message(chat_id, msg)
        
    elif text == "🔄 Scanner Status":
        uptime = format_uptime()
        msg = f"""🔄 <b>Scanner Status</b>

Active: ✅ Running
Scanner Alerts: 0
Total Signals: {cache['total_signals']}
Uptime: {uptime}
Mode: Safe (Rate Limited)"""
        send_message(chat_id, msg)
    
    elif text in ["📊 BTC", "📊 ETH", "📊 XRP", "📊 SOL"]:
        coin = text.split(" ")[1]
        send_message(chat_id, f"{text}\n\nFetching data for {coin}... Coming soon.")
    
    elif text == "💼 Portfolio":
        send_message(chat_id, "💼 <b>Portfolio</b>\n\nFeature coming soon.")
    
    else:
        send_message(chat_id, "Unknown command. Use the buttons below.", get_main_keyboard())
    
    return {"ok": True}

# ==== FOR LOCAL TEST / UPTIME CHECK ====
@app.get("/get_signal")
def get_signal():
    market = fetch_market_data()
    ssi = market["ssi"]
    return JSONResponse({
        "ssi_score": ssi["score"],
        "mood": ssi["mood"], 
        "avg_change": f"{ssi['avg_change']:+.2f}%",
        "strongest": f"{ssi['strongest']} ({ssi['strongest_change']:+.2f}%)"
    })

@app.get("/")
def root():
    return {"message": "CROO Oracle is live"}
