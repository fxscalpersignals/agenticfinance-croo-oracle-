import os
import time
import requests
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from threading import Lock
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CROO Oracle", version="2.0")

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
CACHE_TTL = 300
MAIN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
SYMBOL_TO_NAME = {s: s.replace("USDT", "") for s in MAIN_SYMBOLS}

# ==== CACHE ====
cache = {
    "signals": {},
    "leaderboard": [],
    "last_update": 0,
    "start_time": time.time(),
    "total_scans": 0
}
cache_lock = Lock()

# ==== SET WEBHOOK ON STARTUP ====
@app.on_event("startup")
async def startup_event():
    if TELEGRAM_BOT_TOKEN and RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        try:
            r = requests.post(f"{TELEGRAM_API_URL}/setWebhook", json={"url": webhook_url}, timeout=5)
            print(f"Webhook set: {r.json()}")
        except Exception as e:
            print(f"Webhook failed: {e}")
    run_scanner()

# ==== HEALTH CHECK ====
@app.get("/health")
@app.head("/health")
def health():
    return {
        "status": "ok",
        "agent": "CROO Oracle",
        "assets": list(SYMBOL_TO_NAME.values()),
        "timestamp": int(time.time())
    }

# ==== TA HELPERS ====
def calculate_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))

def calculate_ema(closes: List[float], period: int) -> float:
    if len(closes) < period:
        return closes[-1] if closes else 0
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return np.convolve(closes, weights, mode='valid')[-1]

# ==== BINANCE KLINES ====
def get_klines(symbol: str, interval: str = "1h", limit: int = 100) -> Dict:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5
        )
        if r.status_code!= 200:
            return None
        data = r.json()
        return {
            "closes": [float(x[4]) for x in data],
            "volumes": [float(x[5]) for x in data],
            "highs": [float(x[2]) for x in data],
            "lows": [float(x[3]) for x in data]
        }
    except Exception as e:
        print(f"Binance klines failed for {symbol}: {e}")
        return None

# ==== BINANCE 24HR TICKER ====
def get_binance_24hr(symbol: str) -> float:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": symbol},
            timeout=3
        )
        if r.status_code == 200:
            return float(r.json()["priceChangePercent"])
    except:
        pass
    return 0.0

# ==== MEXC FALLBACK ====
def get_mexc_24hr(symbol: str) -> float:
    try:
        r = requests.get(
            "https://api.mexc.com/api/v3/ticker/24hr",
            params={"symbol": symbol},
            timeout=3
        )
        if r.status_code == 200:
            return float(r.json()["priceChangePercent"])
    except:
        pass
    return 0.0

def get_24hr_change(symbol: str) -> float:
    change = get_binance_24hr(symbol)
    if change!= 0.0:
        return change
    return get_mexc_24hr(symbol)

# ==== PULLBACK ENGINE ====
def analyze_pullback(symbol: str) -> Dict:
    klines = get_klines(symbol, "1h", 100)
    if not klines:
        return {
            "symbol": symbol.replace("USDT", ""),
            "price": 0,
            "signal": "NONE",
            "confidence": 0,
            "entry_zone": "N/A",
            "tp": 0,
            "sl": 0,
            "rsi": 50,
            "reasons": ["❌ API Error"],
            "pullback_pct": 0,
            "rrr": 0,
            "change_24h": 0,
            "ema50": 0
        }
    
    closes = klines["closes"]
    volumes = klines["volumes"]
    highs = klines["highs"]
    
    price = closes[-1]
    rsi = calculate_rsi(closes)
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    
    recent_high = max(highs[-24:]) if len(highs) >= 24 else max(highs)
    pullback_pct = ((recent_high - price) / recent_high) * 100 if recent_high > 0 else 0
    
    avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    vol_spike = volumes[-1] > avg_vol * 1.2 if avg_vol > 0 else False
    
    change_24h = get_24hr_change(symbol)
    
    # ==== CONFIDENCE SCORING ====
    confidence = 0
    reasons = []
    
    if rsi < 40:
        confidence += 25
        reasons.append(f"├── RSI Oversold: +25% ({round(rsi, 1)})")
    elif rsi < 50:
        confidence += 10
        reasons.append(f"├── RSI Cooling: +10% ({round(rsi, 1)})")
    
    if price > ema50:
        confidence += 25
        reasons.append(f"├── Above EMA50: +25% (${round(ema50, 2)})")
    
    if 3 <= pullback_pct <= 10:
        confidence += 25
        reasons.append(f"├── Healthy Pullback: +25% ({round(pullback_pct, 1)}%)")
    elif 1 <= pullback_pct < 3:
        confidence += 10
        reasons.append(f"├── Shallow Pullback: +10% ({round(pullback_pct, 1)}%)")
    
    if vol_spike:
        confidence += 25
        reasons.append(f"└── Volume Spike: +25% ({round(volumes[-1]/avg_vol, 1)}x)")
    
    if confidence >= 75:
        signal = "BUY"
    elif confidence >= 50:
        signal = "WATCH"
    else:
        signal = "NONE"
    
    # ==== ENTRY/TP/SL CALC ====
    entry_low = round(price * 0.995, 2)
    entry_high = round(price * 1.005, 2)
    tp = round(price * 1.045, 2)
    sl = round(ema50 * 0.98, 2) if ema50 > 0 else round(price * 0.96, 2)
    
    risk = price - sl if price > sl else 1
    reward = tp - price
    rrr = round(reward / risk, 2) if risk > 0 else 0
    
    return {
        "symbol": symbol.replace("USDT", ""),
        "price": round(price, 2),
        "change_24h": round(change_24h, 2),
        "signal": signal,
        "confidence": confidence,
        "entry_zone": f"${entry_low} - ${entry_high}",
        "tp": tp,
        "sl": sl,
        "rsi": round(rsi, 1),
        "pullback_pct": round(pullback_pct, 1),
        "rrr": rrr,
        "reasons": reasons if reasons else ["❌ No confluence"],
        "ema50": round(ema50, 2)
    }

# ==== RUN SCANNER ====
def run_scanner():
    now = time.time()
    
    with cache_lock:
        if now - cache["last_update"] < CACHE_TTL:
            return cache
    
    signals = {}
    for symbol in MAIN_SYMBOLS:
        signals[symbol] = analyze_pullback(symbol)
        time.sleep(0.1)
    
    leaderboard = sorted(
        [s for s in signals.values() if s["signal"]!= "NONE"],
        key=lambda x: x["confidence"],
        reverse=True
    )
    
    with cache_lock:
        cache["signals"] = signals
        cache["leaderboard"] = leaderboard
        cache["last_update"] = now
        cache["total_scans"] += 1
    
    print(f"Scanner complete. {len(leaderboard)} signals found.")
    return cache

# ==== CAP ENDPOINT - CROO AGENT API ====
@app.post("/oracle")
async def oracle_api(req: Request):
    try:
        data = await req.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    asset = data.get("asset", "").upper()
    symbol = f"{asset}USDT"
    
    if symbol not in MAIN_SYMBOLS:
        return JSONResponse({
            "error": f"Asset not supported. Use: {list(SYMBOL_TO_NAME.values())}"
        }, status_code=400)
    
    run_scanner()
    signal = cache["signals"].get(symbol, {})
    
    if not signal or signal["signal"] == "NONE":
        return JSONResponse({
            "signal": "NONE",
            "confidence": 0,
            "message": f"No high-confidence pullback for {asset} right now"
        })
    
    return JSONResponse({
        "asset": signal["symbol"],
        "price": signal["price"],
        "signal": signal["signal"],
        "confidence": signal["confidence"],
        "entry_zone": signal["entry_zone"],
        "tp": signal["tp"],
        "sl": signal["sl"],
        "rsi": signal["rsi"],
        "pullback_pct": signal["pullback_pct"],
        "rrr": signal["rrr"],
        "reasons": signal["reasons"]
    })

@app.get("/oracle")
async def oracle_api_get(asset: str = "BTC"):
    symbol = f"{asset.upper()}USDT"
    if symbol not in MAIN_SYMBOLS:
        return JSONResponse({
            "error": f"Asset not supported. Use: {list(SYMBOL_TO_NAME.values())}"
        }, status_code=400)
    
    run_scanner()
    signal = cache["signals"].get(symbol, {})
    
    if not signal or signal["signal"] == "NONE":
        return JSONResponse({
            "signal": "NONE",
            "confidence": 0,
            "message": f"No high-confidence pullback for {asset} right now"
        })
    
    return JSONResponse({
        "asset": signal["symbol"],
        "price": signal["price"],
        "signal": signal["signal"],
        "confidence": signal["confidence"],
        "entry_zone": signal["entry_zone"],
        "tp": signal["tp"],
        "sl": signal["sl"],
        "rsi": signal["rsi"],
        "pullback_pct": signal["pullback_pct"],
        "rrr": signal["rrr"],
        "reasons": signal["reasons"]
    })

# ==== TELEGRAM HELPERS ====
def send_message(chat_id, text, reply_markup=None):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

def get_main_keyboard():
    return {
        "keyboard": [
            [{"text": "📊 Scan All"}, {"text": "📈 Leaderboard"}],
            [{"text": "🔍 BTC"}, {"text": "🔍 ETH"}],
            [{"text": "🔍 XRP"}, {"text": "🔍 SOL"}],
            [{"text": "📊 Status"}]
        ],
        "resize_keyboard": True
    }

def format_uptime():
    seconds = int(time.time() - cache["start_time"])
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs}s"

# ==== TELEGRAM WEBHOOK ====
@app.post("/webhook")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
    except:
        return {"ok": True}
    
    if "message" not in data:
        return {"ok": True}
    
    message = data["message"]
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    
    if text == "/start":
        run_scanner()
        msg = """🚀 <b>CROO Oracle</b>
AI-powered pullback signals for BTC/ETH/XRP/SOL.

<b>Quick Commands:</b>
📊 Scan All — Get all signals
📈 Leaderboard — Rank by confidence
🔍 BTC — Deep dive on specific asset
📊 Status — Scanner health

<i>Not financial advice. Always DYOR.</i>"""
        send_message(chat_id, msg, get_main_keyboard())
    
    elif text == "📊 Scan All":
        run_scanner()
        signals = cache["signals"]
        
        msg = "📊 <b>Pullback Scan Results</b>\n\n"
        found = False
        
        for symbol, s in signals.items():
            if s["signal"]!= "NONE" and s["price"] > 0:
                found = True
                emoji = "🟢" if s["signal"] == "BUY" else "🟡"
                msg += f"{emoji} <b>{s['symbol']}</b>: {s['signal']} ({s['confidence']}%)\n"
                msg += f" Price: ${s['price']} | RSI: {s['rsi']}\n"
                msg += f" Pullback: {s['pullback_pct']}% | R:R {s['rrr']}:1\n\n"
        
        if not found:
            msg += "🔍 No high-confidence pullbacks right now.\nMarket scanning continues 24/7."
        
        send_message(chat_id, msg)
    
    elif text == "📈 Leaderboard":
        run_scanner()
        leaderboard = cache["leaderboard"]
        
        if not leaderboard:
            send_message(chat_id, "📈 No signals yet. Keep watching!")
            return {"ok": True}
        
        msg = "🏆 <b>Signal Leaderboard</b>\n\n"
        for i, s in enumerate(leaderboard[:4], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣"
            msg += f"{medal} <b>{s['symbol']}</b> — {s['confidence']}% {s['signal']}\n"
            msg += f" ${s['price']} | Pullback: {s['pullback_pct']}%\n"
            msg += f" R:R {s['rrr']}:1 | RSI: {s['rsi']}\n\n"
        
        send_message(chat_id, msg)
    
    elif text in ["🔍 BTC", "🔍 ETH", "🔍 XRP", "🔍 SOL"]:
        asset = text.split()[1]
        symbol = f"{asset}USDT"
        run_scanner()
        s = cache["signals"].get(symbol, {})
        
        if not s or s["signal"] == "NONE" or s["price"] == 0:
            send_message(chat_id, f"🔍 No setup for {asset} right now.\n\nUse 📊 Scan All to refresh.")
            return {"ok": True}
        
        reasons_text = "\n".join(s["reasons"])
        msg = f"""🚨 <b>{asset} Pullback Analysis</b>

💰 Price: ${s['price']}
📊 24h Change: {s['change_24h']:+.2f}%

🎯 Signal: <b>{s['signal']}</b>
📈 Confidence: {s['confidence']}%

📉 Pullback: {s['pullback_pct']}% from high
📊 RSI: {s['rsi']}

<b>Entry Zone:</b> {s['entry_zone']}
🎯 TP: ${s['tp']}
🛑 SL: ${s['sl']}
📊 R:R: {s['rrr']}:1

<b>Why:</b>
{reasons_text}

<i>Not financial advice. Always DYOR.</i>"""
        send_message(chat_id, msg)
    
    elif text == "📊 Status":
        run_scanner()
        uptime = format_uptime()
        signal_count = len([s for s in cache["signals"].values() if s["signal"]!= "NONE"])
        
        msg = f"""📊 <b>Scanner Status</b>

✅ Status: Active
📊 Assets: BTC, ETH, XRP, SOL
🔄 Last Scan: Just now
📈 Signals Found: {signal_count}
🔄 Total Scans: {cache['total_scans']}
⏱️ Uptime: {uptime}

<i>Scanning 24/7 for pullback entries</i>"""
        send_message(chat_id, msg)
    
    else:
        send_message(chat_id, "Unknown command. Use the buttons below.", get_main_keyboard())
    
    return {"ok": True}

# ==== ROOT ====
@app.get("/")
def root():
    return {
        "agent": "CROO Oracle",
        "version": "2.0",
        "assets": list(SYMBOL_TO_NAME.values()),
        "capabilities": ["pullback_scan", "confidence_scoring", "telegram_alerts"],
        "api_endpoints": {
            "post": "/oracle",
            "get": "/oracle?asset=BTC",
            "webhook": "/webhook"
        }
    }
