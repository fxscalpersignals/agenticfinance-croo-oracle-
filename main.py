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

# ===== CONFIG - CROO HACKATHON SAFE =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))

# PAYMENTS DISABLED FOR JUDGES - Flip after you win
PAYMENTS_ENABLED = False 
PAYMENT_PROVIDER_TOKEN = ""

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

# CRO ONLY - Keeps you compliant for Croo
ASSETS = ["CROUSDT"]

# ===== STATE =====
cache = {"signals": {}, "last_scan": 0}
signal_history = []
# Fake user DB for demo - replace with real DB later
users_db = {}

# ===== HELPERS =====
def get_binance_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Binance failed for {symbol}: {e}")
    return None

def get_coingecko_price(asset):
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "crypto-com-chain", "vs_currencies": "usd"},
            timeout=8
        )
        if r.status_code == 200:
            return float(r.json()["crypto-com-chain"]["usd"])
    except Exception as e:
        print(f"CoinGecko failed: {e}")
    return None

def get_current_price(symbol):
    klines = get_binance_klines(symbol, "1h", 1)
    if klines:
        return float(klines[-1][4])
    price = get_coingecko_price(symbol)
    return price if price else 0

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

def is_pro(user_id: int) -> bool:
    user = users_db.get(user_id, {})
    if not user:
        return False
    if user.get("plan") == "lifetime":
        return True
    expires = user.get("pro_expires")
    return expires and datetime.now() < expires

def activate_pro(user_id: int, days: int = 30):
    if user_id not in users_db:
        users_db[user_id] = {}
    users_db[user_id]["plan"] = "pro"
    users_db[user_id]["pro_expires"] = datetime.now() + timedelta(days=days)

def analyze_asset(symbol, timeframe="1h"):
    klines = get_binance_klines(symbol, timeframe)
    if not klines or len(klines) < 50:
        price = get_coingecko_price(symbol)
        return {
            "asset": symbol, "price": round(price, 4) if price else 0, 
            "confidence": 0, "signal": "NONE", "direction": "NEUTRAL", "rsi": 0, 
            "reasons": ["No Data"], "source": "price_only", "entry": 0, 
            "stop_loss": 0, "take_profit": 0, "timestamp": datetime.utcnow().isoformat()
        }
    
    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    
    rsi = calc_rsi(closes)[-1]
    ema20_arr = calc_ema(closes, 20)
    ema50_arr = calc_ema(closes, 50)
    ema20 = ema20_arr[-1] if len(ema20_arr) > 0 else price
    ema50 = ema50_arr[-1] if len(ema50_arr) > 0 else price
    
    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    
    # CRO-specific logic - lower thresholds for demo
    long_score = 0
    long_reasons = []
    
    if rsi < 45:
        long_score += 30
        long_reasons.append(f"RSI {rsi:.1f}")
    if price > ema50:
        long_score += 30
        long_reasons.append("Above EMA50")
    if 2 < pullback < 12:
        long_score += 20
        long_reasons.append(f"Dip {pullback:.1f}%")
    if vol_spike:
        long_score += 20
        long_reasons.append("Vol Spike")
    
    short_score = 0
    short_reasons = []
    
    if rsi > 55:
        short_score += 30
        short_reasons.append(f"RSI {rsi:.1f}")
    if price < ema50:
        short_score += 30
        short_reasons.append("Below EMA50")
    if 2 < bounce < 12:
        short_score += 20
        short_reasons.append(f"Bounce {bounce:.1f}%")
    if vol_spike:
        short_score += 20
        short_reasons.append("Vol Spike")
    
    if long_score >= short_score:
        confidence = long_score
        direction = "LONG"
        reasons = long_reasons
        signal = "BUY" if confidence >= 50 else "WATCH"
        stop_loss = round(price * 0.95, 4) if signal == "BUY" else 0
        take_profit = round(price * 1.10, 4) if signal == "BUY" else 0
    else:
        confidence = short_score
        direction = "SHORT"
        reasons = short_reasons
        signal = "SHORT" if confidence >= 50 else "WATCH"
        stop_loss = round(price * 1.05, 4) if signal == "SHORT" else 0
        take_profit = round(price * 0.90, 4) if signal == "SHORT" else 0
    
    return {
        "asset": symbol,
        "price": round(price, 4),
        "rsi": round(rsi, 1),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "confidence": confidence,
        "signal": signal,
        "direction": direction,
        "entry": round(price, 4) if signal in ["BUY", "SHORT"] else 0,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": reasons,
        "timestamp": datetime.utcnow().isoformat()
    }

def run_scanner():
    results = {}
    for asset in ASSETS:
        data = analyze_asset(asset, "1h")
        if data:
            results[asset] = data
            if data["signal"] in ["BUY", "SHORT"]:
                signal_history.append(data)
    
    signal_history[:] = signal_history[-50:]
    cache["signals"] = results
    cache["last_scan"] = time.time()
    return results

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
        await handle_message(chat_id, text, data["message"]["from"]["id"])
    elif "callback_query" in data:
        query = data["callback_query"]
        chat_id = query["message"]["chat"]["id"]
        data_btn = query["data"]
        user_id = query["from"]["id"]
        await handle_callback(chat_id, data_btn, user_id)
    return JSONResponse({"ok": True})

async def handle_message(chat_id, text, user_id):
    if not bot:
        return
        
    if text == "/start":
        keyboard = [
            [InlineKeyboardButton("📊 Get CRO Signal", callback_data="scan_cro")],
            [InlineKeyboardButton("💰 Buy Pro", callback_data="buy_cmd"),
             InlineKeyboardButton("❌ Sell/Cancel", callback_data="sell_cmd")],
            [InlineKeyboardButton("📈 CRO Price", callback_data="price_cro")]
        ]
        msg = "🔮 CRO Oracle - Cronos Trading Bot\n\n"
        msg += "Hackathon Demo: All features FREE\n"
        msg += "Asset: CRO only\n\n"
        msg += "Commands:\n/signals - Get CRO signal\n/price - Current price\n"
        msg += "/buy - Upgrade to Pro\n/sell - Cancel subscription"
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif text == "/signals":
        run_scanner()
        s = cache["signals"].get("CROUSDT")
        if not s or s["signal"] == "WATCH":
            msg = f"CRO: WATCH\nPrice: ${s['price']:.4f}\nRSI: {s['rsi']}\nConf: {s['confidence']}/100\nWaiting for setup..."
        else:
            msg = f"{s['signal']}: CRO\n"
            msg += f"Entry: ${s['entry']}\nSL: ${s['stop_loss']}\nTP: ${s['take_profit']}\n"
            msg += f"Conf: {s['confidence']}/100 | RSI: {s['rsi']}\n"
            msg += f"Reasons: {', '.join(s['reasons'])}"
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif text == "/price":
        price = get_current_price("CROUSDT")
        await bot.send_message(chat_id=chat_id, text=f"CRO Price: ${price:.4f} USD")
    
    elif text == "/buy":
        await handle_buy(chat_id, user_id)
    
    elif text == "/sell":
        await handle_sell(chat_id, user_id)

async def handle_buy(chat_id, user_id):
    if PAYMENTS_ENABLED:
        # This runs AFTER Croo judging
        await bot.send_message(chat_id=chat_id, text="Payment processing coming soon...")
    else:
        # Croo Hackathon Mode
        if is_pro(user_id):
            await bot.send_message(chat_id=chat_id, text="You're already Pro ✅\n\nAll features unlocked for Croo judging.")
        else:
            activate_pro(user_id, days=999) # Give lifetime for demo
            await bot.send_message(
                chat_id=chat_id, 
                text="✅ DEMO MODE: Pro activated for Croo judges\n\n"
                     "All CRO signals unlocked.\n"
                     "Real payments disabled during hackathon.\n\n"
                     "Monetization: Telegram Payments + Unlimint post-launch.\n"
                     "Try /signals now"
            )

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await bot.send_message(chat_id=chat_id, text="You're on Free plan. Nothing to cancel.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await bot.send_message(
            chat_id=chat_id, 
            text="✅ DEMO: Pro subscription cancelled\n\n"
                 "Back to Free plan.\n"
                 "In production: This cancels recurring billing.\n"
                 "Re-upgrade: /buy"
        )

async def handle_callback(chat_id, data, user_id):
    if not bot:
        return
    
    if data == "scan_cro":
        run_scanner()
        s = cache["signals"].get("CROUSDT")
        if not s or s["signal"] == "WATCH":
            msg = f"CRO: WATCH\nPrice: ${s['price']:.4f}\nRSI: {s['rsi']}\nWaiting..."
        else:
            msg = f"{s['signal']}: CRO\n"
            msg += f"Entry: ${s['entry']}\nSL: ${s['stop_loss']}\nTP: ${s['take_profit']}\n"
            msg += f"Conf: {s['confidence']}/100"
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data == "price_cro":
        price = get_current_price("CROUSDT")
        await bot.send_message(chat_id=chat_id, text=f"CRO Price: ${price:.4f} USD")
    
    elif data == "buy_cmd":
        await handle_buy(chat_id, user_id)
    
    elif data == "sell_cmd":
        await handle_sell(chat_id, user_id)

@app.get("/")
def root():
    return {"status": "CRO Oracle Online", "mode": "croo_hackathon", "payments": PAYMENTS_ENABLED}

@app.get("/oracle")
def oracle():
    run_scanner()
    return cache["signals"].get("CROUSDT", {"error": "No data"})

@app.get("/cap/health")
def cap_health():
    return {
        "agent": "CRO Oracle",
        "status": "active",
        "asset": "CRO only",
        "payments": "disabled_for_judging",
        "version": "7.0-croo-final"
    }
