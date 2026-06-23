import os
import time
import requests
import numpy as np
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

app = FastAPI()

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))
CHAT_ID = os.environ.get("CHAT_ID")

PAYMENTS_ENABLED = False
PAYMENT_PROVIDER_TOKEN = ""

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "AVAXUSDT", "DOGEUSDT", "TRXUSDT", "ADAUSDT", "TONUSDT"
]

CG_MAP = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "XRPUSDT": "ripple", "BNBUSDT": "binancecoin", "AVAXUSDT": "avalanche-2",
    "DOGEUSDT": "dogecoin", "TRXUSDT": "tron", "ADAUSDT": "cardano", "TONUSDT": "the-open-network"
}

KRAKEN_MAP = {
    "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XRPUSD", "BNBUSDT": "BNBUSD", "AVAXUSDT": "AVAXUSD",
    "DOGEUSDT": "DOGEUSD", "TRXUSDT": "TRXUSD", "ADAUSDT": "ADAUSD", "TONUSDT": None
}

# ==================== STATE ====================
cache = {"signals": {}, "last_scan": 0, "market_regime": "neutral", "fear_greed": 50}
signal_history = []
users_db = {}
last_alerted = {}
performance = {"wins": 0, "losses": 0, "total": 0}
agent_memory = {
    "last_100_signals": [], "best_asset": "NONE", "best_asset_win_rate": 0.0,
    "total_calls": 0, "revenue_simulated": 0.0
}
scanner_task = None

# ==================== DATA SOURCES ====================
def fetch_coingecko_ohlcv(asset, days=4):
    try:
        coin_id = CG_MAP[asset]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "hourly"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            prices = data["prices"]
            volumes = data["total_volumes"]
            klines = []
            for i in range(len(prices)):
                ts = prices[i][0]
                close = prices[i][1]
                vol = volumes[i][1] if i < len(volumes) else 0
                klines.append([ts, close, close, close, vol])
            return klines[-100:], "CoinGecko"
    except Exception as e:
        print(f"CoinGecko ERR {asset}: {e}")
    return None, None

def fetch_kraken_ohlc(asset):
    try:
        pair = KRAKEN_MAP.get(asset)
        if not pair: return None, None
        url = "https://api.kraken.com/0/public/OHLC"
        r = requests.get(url, params={"pair": pair, "interval": 60}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "error" in data and data["error"]: return None, None
            result_key = list(data["result"].keys())[0]
            return data["result"][result_key][-100:], "Kraken"
    except Exception as e:
        print(f"Kraken ERR {asset}: {e}")
    return None, None

def fetch_coinbase_candles(asset):
    cb_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
              "XRPUSDT": "XRP-USD", "BNBUSDT": "BNB-USD", "AVAXUSDT": "AVAX-USD",
              "DOGEUSDT": "DOGE-USD", "TRXUSDT": "TRX-USD", "ADAUSDT": "ADA-USD"}
    try:
        product = cb_map.get(asset)
        if not product: return None, None
        url = f"https://api.exchange.coinbase.com/products/{product}/candles"
        r = requests.get(url, params={"granularity": 3600}, timeout=10)
        if r.status_code == 200:
            klines = r.json()
            klines.reverse()
            return klines[-100:], "Coinbase"
    except Exception as e:
        print(f"Coinbase ERR {asset}: {e}")
    return None, None

def get_ohlcv(asset):
    for func in [fetch_coingecko_ohlcv, fetch_kraken_ohlc, fetch_coinbase_candles]:
        klines, source = func(asset)
        if klines:
            print(f"{source} OK: {asset}")
            return klines, source
    print(f"ALL SOURCES FAILED: {asset}")
    return None, "none"

def get_current_price(asset):
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
            params={"ids": CG_MAP[asset], "vs_currencies": "usd"}, timeout=8)
        if r.status_code == 200: return float(r.json()[CG_MAP[asset]]["usd"])
    except: pass
    return 0

def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            val = int(r.json()["data"][0]["value"])
            cache["fear_greed"] = val
            return val
    except: pass
    return 50

# ==================== INDICATORS ====================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return np.array([50.0] * len(closes))
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

def calc_ema(prices, period):
    """REAL EMA - not SMA"""
    if len(prices) < period: return np.array([prices[-1]] if len(prices) > 0 else [0])
    alpha = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)

def grade(confidence):
    if confidence >= 90: return "A+"
    elif confidence >= 80: return "A"
    elif confidence >= 70: return "B"
    elif confidence >= 60: return "C"
    elif confidence >= 50: return "D"
    return "F"

# ==================== USER MANAGEMENT ====================
def is_pro(user_id: int) -> bool:
    user = users_db.get(user_id, {})
    if not user: return False
    if user.get("plan") == "lifetime": return True
    expires = user.get("pro_expires")
    return expires and datetime.now() < expires

def activate_pro(user_id: int, days: int = 30):
    if user_id not in users_db: users_db[user_id] = {}
    users_db[user_id]["plan"] = "pro"
    users_db[user_id]["pro_expires"] = datetime.now() + timedelta(days=days)

# ==================== CORE ANALYSIS ====================
def detect_regime():
    klines, _ = get_ohlcv("BTCUSDT")
    if not klines or len(klines) < 50: return "neutral"
    closes = np.array([float(k[4]) for k in klines])
    ema50 = calc_ema(closes, 50)[-1]
    return "bullish" if closes[-1] > ema50 else "bearish"

def analyze_asset(symbol):
    klines, source = get_ohlcv(symbol)
    if not klines or len(klines) < 50:
        price = get_current_price(symbol)
        if price > 0:
            return {
                "asset": symbol.replace("USDT", ""), "signal": "WATCH", "confidence": 20,
                "grade": "F", "price": round(price, 4), "entry": round(price, 4),
                "stop_loss": round(price * 0.97, 4), "take_profit": round(price * 1.05, 4),
                "bullish_reasons": ["Price only"], "bearish_reasons": [],
                "missing_conditions": ["Full OHLCV data unavailable"], "source": "price_only", "direction": "NONE"
            }
        return {"asset": symbol.replace("USDT", ""), "signal": "NONE", "confidence": 0, "price": 0, "bullish_reasons": ["No Data"], "bearish_reasons": [], "direction": "NONE"}

    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    rsi_val = calc_rsi(closes)[-1]
    ema20 = calc_ema(closes, 20)[-1]
    ema50 = calc_ema(closes, 50)[-1]

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    price_near_ema20 = abs(price - ema20) / ema20 * 100 < 1.5
    bullish_confirmation = price > prev_close
    bearish_confirmation = price < prev_close

    long_score = 0
    short_score = 0
    bullish_reasons = []
    bearish_reasons = []
    missing_conditions = []

    # 1. RSI: 20 pts
    if rsi_val < 45:
        long_score += 20
        bullish_reasons.append("RSI Oversold")
    elif rsi_val > 55:
        short_score += 20
        bearish_reasons.append("RSI Overbought")
    else:
        missing_conditions.append("RSI neutral")

    # 2. EMA Trend: 20 pts
    if price > ema50:
        long_score += 20
        bullish_reasons.append("Above EMA50")
    elif price < ema50:
        short_score += 20
        bearish_reasons.append("Below EMA50")
    else:
        missing_conditions.append("No clear EMA trend")

    # 3. Pullback + Support: 20 pts - FIXED LOGIC
    if price > ema50 and 4 < pullback < 12 and price_near_ema20:
        long_score += 20
        bullish_reasons.append(f"Meaningful Dip {pullback:.1f}% to EMA20")
    elif price < ema50 and 4 < bounce < 12 and price_near_ema20:
        short_score += 20
        bearish_reasons.append(f"Dead Cat Bounce {bounce:.1f}% to EMA20")
    else:
        missing_conditions.append("Pullback too shallow/deep or not at EMA20")

    # 4. Volume Spike: 20 pts
    if vol_spike:
        if long_score >= short_score:
            long_score += 20
            bullish_reasons.append("Volume Spike")
        else:
            short_score += 20
            bearish_reasons.append("Volume Spike")
    else:
        missing_conditions.append("No volume confirmation")

    # 5. Confirmation Candle: 20 pts
    if bullish_confirmation:
        long_score += 20
        bullish_reasons.append("Bullish Confirmation Candle")
    elif bearish_confirmation:
        short_score += 20
        bearish_reasons.append("Bearish Confirmation Candle")
    else:
        missing_conditions.append("No confirmation candle")

    # Fear & Greed bonus
    fg = cache["fear_greed"]
    if fg < 25 and long_score >= short_score:
        long_score += 5
        bullish_reasons.append("Extreme Fear")
    if fg > 75 and short_score > long_score:
        short_score += 5
        bearish_reasons.append("Extreme Greed")

    direction = "LONG" if long_score >= short_score else "SHORT"
    confidence = max(long_score, short_score)

    # NEW THRESHOLDS: BUY/SHORT >= 70, WATCH >= 50
    signal = "NONE"
    if confidence >= 70: signal = "BUY" if direction == "LONG" else "SHORT"
    elif confidence >= 50: signal = "WATCH"

    # WATCH now shows actionable levels
    if signal == "BUY":
        stop_loss = round(price * 0.95, 4)
        take_profit = round(price * 1.10, 4)
        entry = round(price, 4)
    elif signal == "SHORT":
        stop_loss = round(price * 1.05, 4)
        take_profit = round(price * 0.90, 4)
        entry = round(price, 4)
    elif signal == "WATCH":
        entry = round(price, 4)
        if direction == "LONG":
            stop_loss = round(price * 0.97, 4)
            take_profit = round(price * 1.05, 4)
        else:
            stop_loss = round(price * 1.03, 4)
            take_profit = round(price * 0.95, 4)
    else:
        stop_loss = 0
        take_profit = 0
        entry = 0

    if not bullish_reasons: bullish_reasons = ["Waiting for setup"]
    if not bearish_reasons: bearish_reasons = ["Waiting for setup"]

    return {
        "asset": symbol.replace("USDT", ""), "price": round(price, 4), "signal": signal,
        "confidence": confidence, "grade": grade(confidence), "direction": direction,
        "entry": entry, "stop_loss": stop_loss, "take_profit": take_profit,
        "rsi": round(rsi_val, 1), "bullish_reasons": bullish_reasons,
        "bearish_reasons": bearish_reasons, "missing_conditions": missing_conditions,
        "source": source, "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"], "timestamp": datetime.utcnow().isoformat()
    }

def update_performance():
    for signal in signal_history:
        if signal.get("status") == "open" and signal.get("entry", 0) > 0:
            current = get_current_price(signal["asset"] + "USDT")
            if current == 0: continue
            if signal["direction"] == "LONG":
                if current >= signal["take_profit"]:
                    signal["pnl"] = round((signal["take_profit"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"; performance["wins"] += 1; performance["total"] += 1
                elif current <= signal["stop_loss"]:
                    signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"; performance["losses"] += 1; performance["total"] += 1
            elif signal["direction"] == "SHORT":
                if current <= signal["take_profit"]:
                    signal["pnl"] = round((signal["entry"] - signal["take_profit"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"; performance["wins"] += 1; performance["total"] += 1
                elif current >= signal["stop_loss"]:
                    signal["pnl"] = round((signal["entry"] - signal["stop_loss"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"; performance["losses"] += 1; performance["total"] += 1
    update_memory()

def update_memory():
    agent_memory["last_100_signals"] = signal_history[-100:]
    stats = {}
    for sig in signal_history:
        if sig.get("status") in ["win", "loss"]:
            asset = sig["asset"]
            if asset not in stats: stats[asset] = {"wins": 0, "total": 0}
            stats[asset]["total"] += 1
            if sig["status"] == "win": stats[asset]["wins"] += 1
    best = None
    best_rate = 0
    for asset, s in stats.items():
        if s["total"] >= 3:
            rate = s["wins"] / s["total"]
            if rate > best_rate:
                best_rate = rate; best = asset
    agent_memory["best_asset"] = best or "NONE"
    agent_memory["best_asset_win_rate"] = round(best_rate * 100, 1)

async def send_alert(signal):
    if not bot or signal["confidence"] < 70: return
    if signal["asset"] in last_alerted and time.time() - last_alerted[signal["asset"]] < 3600: return
    msg = f"🚨 NEW {signal['signal']} SIGNAL\n\n"
    msg += f"Asset: {signal['asset']}\nConfidence: {signal['confidence']}% ({signal['grade']})\n\n"
    msg += f"Entry:\n{signal['entry']}\n\nTarget:\n{signal['take_profit']}\n\n"
    msg += f"Stop:\n{signal['stop_loss']}\n\nReasons:\n" + "\n".join([f"✅ {r}" for r in (signal['bullish_reasons'] if signal['direction']=='LONG' else signal['bearish_reasons'])])
    msg += f"\n\nMarket: {signal['market_regime'].upper()} | F&G: {signal['fear_greed']} | Source: {signal['source']}"
    if CHAT_ID:
        try: await bot.send_message(chat_id=CHAT_ID, text=msg)
        except: pass
    last_alerted[signal["asset"]] = time.time()

def scan_all():
    print(f"AUTO SCAN {datetime.utcnow()}")
    fetch_fear_greed()
    update_performance()
    cache["market_regime"] = detect_regime()
    results = {}
    for asset in ASSETS:
        data = analyze_asset(asset)
        if data:
            results[asset] = data
            if data["signal"] in ["BUY", "SHORT"]:
                data["status"] = "open"
                signal_history.append(data)
                agent_memory["total_calls"] += 1
                agent_memory["revenue_simulated"] += 0.01
                asyncio.create_task(send_alert(data))
    signal_history[:] = signal_history[-100:]
    cache["signals"] = results
    cache["last_scan"] = time.time()
    print(f"Scan complete. Signals: {len(results)}")
    return results

async def scanner_loop():
    global scanner_task
    print("Auto scanner started")
    while True:
        try:
            scan_all()
            await asyncio.sleep(300)
        except Exception as e:
            print(f"Scanner error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    global scanner_task
    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
        print(f"Webhook set: {webhook_url}")
    scan_all()
    scanner_task = asyncio.create_task(scanner_loop())

# ==================== TELEGRAM ====================
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        user_id = data["message"]["from"]["id"]
        await handle_message(chat_id, text, user_id)
    elif "callback_query" in data:
        query = data["callback_query"]
        await handle_callback(query["message"]["chat"]["id"], query["data"], query["from"]["id"])
    return JSONResponse({"ok": True})

async def handle_message(chat_id, text, user_id):
    if not bot: return
    if text == "/start":
        scan_all()
        keyboard = [
            [InlineKeyboardButton("📊 Scan Markets", callback_data="scan_all"),
             InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("🔍 Best Signal", callback_data="best_signal")],
            [InlineKeyboardButton("📈 BTC", callback_data="BTCUSDT"),
             InlineKeyboardButton("📈 ETH", callback_data="ETHUSDT"),
             InlineKeyboardButton("📈 SOL", callback_data="SOLUSDT")],
            [InlineKeyboardButton("📈 BNB", callback_data="BNBUSDT"),
             InlineKeyboardButton("📈 XRP", callback_data="XRPUSDT")],
            [InlineKeyboardButton("📈 AVAX", callback_data="AVAXUSDT"),
             InlineKeyboardButton("📈 DOGE", callback_data="DOGEUSDT")],
            [InlineKeyboardButton("📈 TRX", callback_data="TRXUSDT"),
             InlineKeyboardButton("📈 ADA", callback_data="ADAUSDT")],
            [InlineKeyboardButton("📈 TON", callback_data="TONUSDT"),
             InlineKeyboardButton("💎 Upgrade", callback_data="buy_cmd")]
        ]
        regime = cache["market_regime"].upper()
        signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
        top = signals[0] if signals else None

        msg = "🔮 CROO AI Oracle\n"
        msg += f"Market: {regime} | F&G: {cache['fear_greed']}\n"
        msg += f"Assets: {len(ASSETS)} monitored\n"
        if top and top.get("confidence", 0) > 0:
            msg += f"\n🔥 Top: {top.get('asset')} {top.get('signal')} {top.get('confidence')}% ({top.get('grade')})\n"
            msg += f"Price: ${top.get('price')} | {top.get('source', 'N/A')}\n"
        msg += "\n/scan /best /leaderboard /stats"
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))
    elif text in ["/scan", "/signals"]:
        scan_all(); await send_leaderboard(chat_id)
    elif text == "/best":
        scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            await bot.send_message(chat_id=chat_id, text="No signals yet. Scanning...")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)))
    elif text == "/leaderboard": scan_all(); await send_leaderboard(chat_id)
    elif text == "/stats": await send_stats(chat_id)
    elif text == "/buy": await handle_buy(chat_id, user_id)
    elif text == "/sell": await handle_sell(chat_id, user_id)

async def send_rich_card(chat_id, s):
    if s.get('signal') == "WATCH":
        msg = f"⚠️ WATCH SIGNAL\n\n"
        msg += f"Asset: {s.get('asset')}\nConfidence: {s.get('confidence')}% ({s.get('grade')})\nPrice: ${s.get('price')}\n\n"
        msg += f"Entry Zone: ${s.get('entry')}\nTarget: ${s.get('take_profit')}\nStop: ${s.get('stop_loss')}\n\n"
    else:
        msg = f"🚨 {s.get('signal')} SIGNAL\n\n"
        msg += f"Asset: {s.get('asset')}\nConfidence: {s.get('confidence')}% ({s.get('grade')})\nPrice: ${s.get('price')}\n\n"
        msg += f"Entry:\n${s.get('entry')}\n\nTarget:\n${s.get('take_profit')}\n\nStop:\n${s.get('stop_loss')}\n\n"

    if s.get('direction') == 'LONG':
        msg += f"Bullish Reasons:\n" + "\n".join([f"✅ {r}" for r in s.get('bullish_reasons', ['None'])])
    else:
        msg += f"Bearish Reasons:\n" + "\n".join([f"✅ {r}" for r in s.get('bearish_reasons', ['None'])])

    if s.get('missing_conditions'):
        msg += f"\n\nMissing Conditions:\n" + "\n".join([f"❌ {m}" for m in s.get('missing_conditions')])
    msg += f"\n\nNeed 70% for BUY/SHORT\nCurrent: {s.get('confidence')}%"
    msg += f"\nMarket: {s.get('market_regime','').upper()} | F&G: {s.get('fear_greed')} | Source: {s.get('source', 'N/A')}"
    await bot.send_message(chat_id=chat_id, text=msg)

async def send_leaderboard(chat_id):
    scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    msg = f"🏆 LEADERBOARD | {cache['market_regime'].upper()} | F&G: {cache['fear_greed']}\n\n"
    for i, s in enumerate(signals[:10], 1):
        msg += f"{i}. {s.get('asset','N/A')} - {s.get('confidence',0)}% ({s.get('grade','N/A')}) {s.get('signal','NONE')}\n"
        msg += f" ${s.get('price',0)} | {s.get('source','N/A')}\n"
    await bot.send_message(chat_id=chat_id, text=msg)

async def send_stats(chat_id):
    update_performance()
    win_rate = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0
    msg = f"📊 AGENT STATS\n\n"
    msg += f"Total Signals: {performance['total']}\n"
    msg += f"Wins: {performance['wins']}\n"
    msg += f"Losses: {performance['losses']}\n"
    msg += f"Win Rate: {win_rate}%\n"
    msg += f"Market Regime: {cache['market_regime'].upper()}\n"
    msg += f"Fear & Greed: {cache['fear_greed']}\n"
    msg += f"Best Asset: {agent_memory['best_asset']} ({agent_memory['best_asset_win_rate']}%)"
    await bot.send_message(chat_id=chat_id, text=msg)

async def handle_buy(chat_id, user_id):
    if PAYMENTS_ENABLED:
        await bot.send_message(chat_id=chat_id, text="Payments enabled post-launch.")
    else:
        if is_pro(user_id):
            await bot.send_message(chat_id=chat_id, text="You're already Pro ✅")
        else:
            activate_pro(user_id, 999)
            await bot.send_message(chat_id=chat_id, text="✅ Pro activated\nAll signals unlocked.")

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await bot.send_message(chat_id=chat_id, text="You're on Free plan.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await bot.send_message(chat_id=chat_id, text="✅ Pro cancelled")

async def handle_callback(chat_id, data, user_id):
    if data == "scan_all": scan_all(); await send_leaderboard(chat_id)
    elif data == "best_signal":
        scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if signals: await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)))
        else: await bot.send_message(chat_id=chat_id, text="Scanning... try again in 10s")
    elif data == "leaderboard": scan_all(); await send_leaderboard(chat_id)
    elif data == "buy_cmd": await handle_buy(chat_id, user_id)
    elif data == "sell_cmd": await handle_sell(chat_id, user_id)
    elif data in ASSETS:
        scan_all()
        s = cache["signals"].get(data)
        if s: await send_rich_card(chat_id, s)
        else: await bot.send_message(chat_id=chat_id, text=f"No data for {data.replace('USDT','')}. All sources failed.")

# ==================== API ENDPOINTS ====================
@app.get("/")
def root():
    return {"status": "CROO AI Oracle Online", "version": "10.1"}

@app.get("/health")
def health():
    return {"status": "ok", "uptime": int(time.time() - start_time)}

@app.get("/oracle")
def oracle():
    scan_all()
    return cache["signals"]

@app.get("/best_signal")
def best_signal():
    scan_all()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    if not signals: return {"asset": "NONE", "signal": "NONE", "confidence": 0}
    best = max(signals, key=lambda x: x.get("confidence", 0))
    return {"asset": best.get("asset"), "signal": best.get("signal"), "confidence": best.get("confidence"),
            "grade": best.get("grade"), "entry": best.get("entry"), "price": best.get("price"), "source": best.get("source")}

@app.get("/leaderboard")
def leaderboard():
    scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    return [{"asset": s.get("asset"), "confidence": s.get("confidence"), "grade": s.get("grade"),
             "signal": s.get("signal"), "price": s.get("price"), "source": s.get("source")} for s in signals]

@app.get("/performance")
def get_performance():
    update_performance()
    win_rate = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0
    return {"total_signals": performance["total"], "wins": performance["wins"],
            "losses": performance["losses"], "win_rate": f"{win_rate}%"}

@app.post("/agent/query")
async def agent_query(request: Request):
    data = await request.json()
    task = data.get("task", "")
    scan_all()
    if task == "find_best_pullback":
        signals = [s for s in cache["signals"].values() if "Dip" in str(s.get("bullish_reasons", []))]
        if not signals: return {"error": "No pullback found"}
        best = max(signals, key=lambda x: x.get("confidence", 0))
        return {"asset": best.get("asset"), "confidence": best.get("confidence"), "signal": best.get("signal"), "grade": best.get("grade")}
    return best_signal()

@app.get("/history")
def history():
    return signal_history[-50:]

@app.get("/.well-known/agent.json")
def agent_manifest():
    return {
        "name": "CROO AI Oracle",
        "description": "Autonomous crypto intelligence agent",
        "version": "10.1",
        "endpoint": "/agent/query",
        "capabilities": ["pullback_detection", "market_intelligence", "signal_ranking", "regime_detection", "explainability"]
    }

@app.get("/explain/{symbol}")
def explain(symbol: str):
    scan_all()
    asset = symbol.upper() + "USDT"
    signal = cache["signals"].get(asset)
    if not signal: return {"error": "No signal found", "symbol": symbol}
    return {
        "asset": signal.get("asset"), "signal": signal.get("signal"), "confidence": signal.get("confidence"),
        "grade": signal.get("grade"), "bullish_reasons": signal.get("bullish_reasons"),
        "bearish_reasons": signal.get("bearish_reasons"), "missing_conditions": signal.get("missing_conditions"),
        "market_regime": signal.get("market_regime"), "fear_greed": signal.get("fear_greed"),
        "price": signal.get("price"), "rsi": signal.get("rsi"), "source": signal.get("source")
    }

@app.get("/agent/revenue")
def revenue():
    return {
        "model": "pay_per_signal", "price_per_call": "0.01 CRO", "monthly_projection": "500 CRO",
        "total_calls": agent_memory["total_calls"], "revenue_simulated": round(agent_memory["revenue_simulated"], 2)
    }

@app.get("/stats")
def stats():
    update_performance()
    accuracy = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0
    return {
        "accuracy": f"{accuracy}%", "total_signals": performance["total"], "wins": performance["wins"],
        "losses": performance["losses"], "market_regime": cache["market_regime"], "fear_greed": cache["fear_greed"],
        "best_asset": agent_memory["best_asset"], "best_asset_win_rate": f"{agent_memory['best_asset_win_rate']}%"
    }

@app.get("/reputation")
def reputation():
    score = min(100, performance["wins"] * 2)
    return {"reputation_score": score, "grade": grade(score)}

@app.get("/agent/memory")
def get_memory():
    return agent_memory

@app.get("/demo")
def demo():
    scan_all()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    if not signals: return {"error": "No signals"}
    best = max(signals, key=lambda x: x.get("confidence", 0))
    return {
        "best_signal": best.get("asset"), "confidence": best.get("confidence"), "grade": best.get("grade"),
        "entry": best.get("entry"), "tp": best.get("take_profit"), "sl": best.get("stop_loss"),
        "market_regime": best.get("market_regime"), "fear_greed": best.get("fear_greed"),
        "signal": best.get("signal"), "source": best.get("source")
    }

@app.get("/cap/metadata")
def cap_metadata():
    return {"agent": "CRO
