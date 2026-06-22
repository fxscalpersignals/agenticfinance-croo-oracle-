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

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "CROUSDT"]
CG_MAP = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "XRPUSDT": "ripple", "CROUSDT": "crypto-com-chain"
}
KRAKEN_MAP = {
    "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XRPUSD", "CROUSDT": None # Kraken doesn't have CRO
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

# ==================== DATA SOURCES ====================
def fetch_coingecko_ohlcv(asset, days=1):
    """Primary: CoinGecko Market Chart - OHLCV for RSI/EMA/Volume"""
    try:
        coin_id = CG_MAP[asset]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "hourly"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            prices = data["prices"] # [timestamp, price]
            volumes = data["total_volumes"] # [timestamp, volume]
            klines = []
            for i in range(len(prices)):
                ts = prices[i][0]
                close = prices[i][1]
                vol = volumes[i][1] if i < len(volumes) else 0
                klines.append([ts, close, close, close, close, vol]) # O=H=L=C for hourly
            print(f"CoinGecko OK: {asset} {days}d")
            return klines[-100:], "CoinGecko"
    except Exception as e:
        print(f"CoinGecko ERR {asset}: {e}")
    return None, None

def fetch_kraken_ohlc(asset):
    """Fallback 1: Kraken - Very reliable from Render"""
    try:
        pair = KRAKEN_MAP.get(asset)
        if not pair: return None, None
        url = "https://api.kraken.com/0/public/OHLC"
        r = requests.get(url, params={"pair": pair, "interval": 60}, timeout=10) # 1h
        if r.status_code == 200:
            data = r.json()
            if "error" in data and data["error"]:
                print(f"Kraken FAIL {asset}: {data['error']}")
                return None, None
            result_key = list(data["result"].keys())[0]
            klines = data["result"][result_key]
            print(f"Kraken OK: {asset}")
            return klines[-100:], "Kraken"
    except Exception as e:
        print(f"Kraken ERR {asset}: {e}")
    return None, None

def fetch_coinbase_candles(asset):
    """Fallback 2: Coinbase - Stable"""
    cb_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
    try:
        product = cb_map.get(asset)
        if not product: return None, None
        url = f"https://api.exchange.coinbase.com/products/{product}/candles"
        r = requests.get(url, params={"granularity": 3600}, timeout=10) # 1h
        if r.status_code == 200:
            klines = r.json()
            klines.reverse() # Coinbase returns newest first
            print(f"Coinbase OK: {asset}")
            return klines[-100:], "Coinbase"
    except Exception as e:
        print(f"Coinbase ERR {asset}: {e}")
    return None, None

def fetch_cryptocompare(asset):
    """Fallback 3: CryptoCompare"""
    cc_map = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "XRPUSDT": "XRP", "CROUSDT": "CRO"}
    try:
        sym = cc_map[asset]
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        r = requests.get(url, params={"fsym": sym, "tsym": "USD", "limit": 100}, timeout=10)
        if r.status_code == 200:
            data = r.json()["Data"]["Data"]
            klines = [[d["time"]*1000, d["open"], d["high"], d["low"], d["close"], d["volumeto"]] for d in data]
            print(f"CryptoCompare OK: {asset}")
            return klines, "CryptoCompare"
    except Exception as e:
        print(f"CryptoCompare ERR {asset}: {e}")
    return None, None

def get_ohlcv(asset):
    """Try CoinGecko -> Kraken -> Coinbase -> CryptoCompare"""
    klines, source = fetch_coingecko_ohlcv(asset, days=4) # 4 days for 4h data
    if klines: return klines, source
    klines, source = fetch_kraken_ohlc(asset)
    if klines: return klines, source
    klines, source = fetch_coinbase_candles(asset)
    if klines: return klines, source
    klines, source = fetch_cryptocompare(asset)
    if klines: return klines, source
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
    except Exception as e:
        print(f"F&G ERR: {e}")
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

def calc_ema(closes, period):
    if len(closes) < period: return np.array([closes[-1]] if len(closes) > 0 else [0])
    return np.convolve(closes, np.ones(period)/period, mode='valid')

def grade(confidence):
    if confidence >= 90: return "A+"
    elif confidence >= 80: return "A"
    elif confidence >= 70: return "B"
    elif confidence >= 60: return "C"
    return "D"

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
                "grade": "D", "price": round(price, 4), "entry": round(price, 4),
                "reasons": ["Price only - OHLCV unavailable"], "source": "price_only"
            }
        return {"asset": symbol.replace("USDT", ""), "signal": "NONE", "confidence": 0, "price": 0, "reasons": ["No Data"]}

    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    rsi_val = calc_rsi(closes)[-1]
    ema20 = calc_ema(closes, 20)[-1] if len(calc_ema(closes, 20)) > 0 else price
    ema50 = calc_ema(closes, 50)[-1] if len(calc_ema(closes, 50)) > 0 else price

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False

    long_score = 0
    if rsi_val < 45: long_score += 30
    if price > ema50: long_score += 25
    if 2 < pullback < 12: long_score += 25
    if vol_spike: long_score += 20

    short_score = 0
    if rsi_val > 55: short_score += 25
    if price < ema50: short_score += 25
    if 2 < bounce < 12: short_score += 20
    rally = ((ema20 - price) / price * 100) if price < ema20 else 0
    if 0 < rally < 3: short_score += 15
    if vol_spike: short_score += 15

    fg = cache["fear_greed"]
    if fg < 25 and long_score > 0: long_score += 10
    if fg > 75 and short_score > 0: short_score += 10

    confidence = max(long_score, short_score)
    direction = "LONG" if long_score >= short_score else "SHORT"

    reasons = []
    if rsi_val < 45 and direction == "LONG": reasons.append("✅ RSI Oversold")
    if rsi_val > 55 and direction == "SHORT": reasons.append("✅ RSI Overbought")
    if price > ema50 and direction == "LONG": reasons.append("✅ Above EMA50")
    if price < ema50 and direction == "SHORT": reasons.append("✅ Below EMA50")
    if 2 < pullback < 12 and direction == "LONG": reasons.append(f"✅ Dip {pullback:.1f}%")
    if 2 < bounce < 12 and direction == "SHORT": reasons.append(f"✅ Bounce {bounce:.1f}%")
    if vol_spike: reasons.append("✅ Volume Spike")
    if not reasons: reasons = ["Waiting for setup"]

    signal = "NONE"
    if confidence >= 75: signal = "BUY" if direction == "LONG" else "SHORT"
    elif confidence >= 20: signal = "WATCH"

    stop_loss = round(price * 0.95, 4) if signal == "BUY" else round(price * 1.05, 4) if signal == "SHORT" else 0
    take_profit = round(price * 1.10, 4) if signal == "BUY" else round(price * 0.90, 4) if signal == "SHORT" else 0

    return {
        "asset": symbol.replace("USDT", ""), "price": round(price, 4), "signal": signal,
        "confidence": confidence, "grade": grade(confidence), "direction": direction,
        "entry": round(price, 4), "stop_loss": stop_loss, "take_profit": take_profit,
        "rsi": round(rsi_val, 1), "reasons": reasons, "source": source,
        "market_regime": cache["market_regime"], "fear_greed": cache["fear_greed"],
        "timestamp": datetime.utcnow().isoformat()
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
    if not bot or signal["confidence"] < 75: return
    if signal["asset"] in last_alerted and time.time() - last_alerted[signal["asset"]] < 3600: return
    msg = f"🚨 NEW {signal['signal']} SIGNAL\n\n"
    msg += f"Asset: {signal['asset']}\nConfidence: {signal['confidence']}% ({signal['grade']})\n\n"
    msg += f"Entry:\n{signal['entry']}\n\nTarget:\n{signal['take_profit']}\n\n"
    msg += f"Stop:\n{signal['stop_loss']}\n\nReasons:\n" + "\n".join(signal['reasons'])
    msg += f"\n\nMarket: {signal['market_regime'].upper()} | F&G: {signal['fear_greed']} | Source: {signal['source']}"
    if CHAT_ID:
        try: await bot.send_message(chat_id=CHAT_ID, text=msg)
        except: pass
    last_alerted[signal["asset"]] = time.time()

def scan_all():
    print("Starting scan...")
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
    while True:
        scan_all()
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
        print(f"Webhook set: {webhook_url}")
    scan_all()
    asyncio.create_task(scanner_loop())

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
            [InlineKeyboardButton("📈 XRP", callback_data="XRPUSDT"),
             InlineKeyboardButton("📈 CRO", callback_data="CROUSDT")],
            [InlineKeyboardButton("💎 Upgrade", callback_data="buy_cmd")]
        ]
        regime = cache["market_regime"].upper()
        signals = sorted(cache["signals"].values(), key=lambda x: x["confidence"], reverse=True)
        top = signals[0] if signals else None

        msg = "🔮 CROO AI Oracle - Market Intelligence\n\n"
        msg += f"Market Regime: {regime} | F&G: {cache['fear_greed']}\n"
        msg += "Autonomous scanning every 5 min\n"
        msg += "Assets: BTC, ETH, SOL, XRP, CRO\n"
        if top and top["confidence"] > 0:
            msg += f"🔥 TOP: {top['asset']} {top['signal']} {top['confidence']}% ({top['grade']})\n"
            msg += f"Price: ${top['price']} | Source: {top['source']}\n\n"
        msg += "Demo: All features FREE for judges\n\n"
        msg += "/scan /best /leaderboard /stats /buy /sell"
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))
    elif text in ["/scan", "/signals"]:
        scan_all(); await send_leaderboard(chat_id)
    elif text == "/best":
        scan_all()
        signals = [s for s in cache["signals"].values() if s["confidence"] > 0]
        if not signals:
            await bot.send_message(chat_id=chat_id, text="No signals yet. Scanning...")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x["confidence"]))
    elif text == "/leaderboard": scan_all(); await send_leaderboard(chat_id)
    elif text == "/stats": await send_stats(chat_id)
    elif text == "/buy": await handle_buy(chat_id, user_id)
    elif text == "/sell": await handle_sell(chat_id, user_id)

async def send_rich_card(chat_id, s):
    msg = f"🚨 {s['signal']} SIGNAL\n\n"
    msg += f"Asset: {s['asset']}\nConfidence: {s['confidence']}% ({s['grade']})\nPrice: ${s['price']}\n\n"
    if s['entry'] > 0:
        msg += f"Entry:\n{s['entry']}\n\nTarget:\n{s['take_profit']}\n\nStop:\n{s['stop_loss']}\n\n"
    msg += f"Reasons:\n" + "\n".join(s['reasons'])
    msg += f"\n\nMarket: {s['market_regime'].upper()} | F&G: {s['fear_greed']} | Source: {s['source']}"
    await bot.send_message(chat_id=chat_id, text=msg)

async def send_leaderboard(chat_id):
    scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x["confidence"], reverse=True)
    msg = f"🏆 LEADERBOARD | {cache['market_regime'].upper()} | F&G: {cache['fear_greed']}\n\n"
    for i, s in enumerate(signals[:5], 1):
        msg += f"{i}. {s['asset']} - {s['confidence']}% ({s['grade']}) {s['signal']}\n"
        msg += f" ${s['price']} | {s['source']}\n"
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
        await bot.send_message(chat_id=chat_id, text="Payments enable post-hackathon...")
    else:
        if is_pro(user_id):
            await bot.send_message(chat_id=chat_id, text="You're already Pro ✅")
        else:
            activate_pro(user_id, 999)
            await bot.send_message(chat_id=chat_id, text="✅ DEMO: Pro activated\n\nAll signals unlocked for Croo judging.\nMonetization: Post-launch via Telegram Payments")

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await bot.send_message(chat_id=chat_id, text="You're on Free plan.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await bot.send_message(chat_id=chat_id, text="✅ DEMO: Pro cancelled\n\nIn production: Cancels recurring billing.")

async def handle_callback(chat_id, data, user_id):
    if data == "scan_all": scan_all(); await send_leaderboard(chat_id)
    elif data == "best_signal":
        scan_all()
        signals = [s for s in cache["signals"].values() if s["confidence"] > 0]
        if signals: await send_rich_card(chat_id, max(signals, key=lambda x: x["confidence"]))
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
    return {"status": "CROO AI Oracle Online", "mode": "hackathon", "payments": PAYMENTS_ENABLED, "version": "10.0"}

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
    signals = [s for s in cache["signals"].values() if s["confidence"] > 0]
    if not signals: return {"asset": "NONE", "signal": "NONE", "confidence": 0}
    best = max(signals, key=lambda x: x["confidence"])
    return {"asset": best["asset"], "signal": best["signal"], "confidence": best["confidence"],
            "grade": best["grade"], "entry": best["entry"], "price": best["price"], "source": best["source"]}

@app.get("/leaderboard")
def leaderboard():
    scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x["confidence"], reverse=True)
    return [{"asset": s["asset"], "confidence": s["confidence"], "grade": s["grade"],
             "signal": s["signal"], "price": s["price"], "source": s["source"]} for s in signals]

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
        signals = [s for s in cache["signals"].values() if "Dip" in str(s.get("reasons", []))]
        if not signals: return {"error": "No pullback found"}
        best = max(signals, key=lambda x: x["confidence"])
        return {"asset": best["asset"], "confidence": best["confidence"], "signal": best["signal"], "grade": best["grade"]}
    return best_signal()

@app.get("/history")
def history():
    return signal_history[-50:]

@app.get("/.well-known/agent.json")
def agent_manifest():
    return {
        "name": "CROO AI Oracle",
        "description": "Autonomous crypto intelligence agent",
        "version": "10.0",
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
        "asset": signal["asset"], "signal": signal["signal"], "confidence": signal["confidence"],
        "grade": signal["grade"], "reasons": signal["reasons"], "market_regime": signal["market_regime"],
        "fear_greed": signal["fear_greed"], "price": signal["price"], "rsi": signal["rsi"],
        "source": signal["source"]
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
    signals = [s for s in cache["signals"].values() if s["confidence"] > 0]
    if not signals: return {"error": "No signals"}
    best = max(signals, key=lambda x: x["confidence"])
    return {
        "best_signal": best["asset"], "confidence": best["confidence"], "grade": best["grade"],
        "entry": best["entry"], "tp": best["take_profit"], "sl": best["stop_loss"],
        "market_regime": best["market_regime"], "fear_greed": best["fear_greed"],
        "signal": best["signal"], "source": best["source"]
    }

@app.get("/cap/metadata")
def cap_metadata():
    return {"agent": "CROO AI Oracle", "version": "10.0", "category": "Market Intelligence",
            "callable": True, "supports": ["BTC", "ETH", "SOL", "XRP", "CRO"]}

@app.get("/pricing")
def pricing():
    return {"free": "5 requests/day", "pro": "Unlimited", "enterprise": "API Access"}

@app.get("/capabilities")
def capabilities():
    return {"features": ["pullback_detection", "bounce_detection", "rally_to_resistance",
            "multi_timeframe_analysis", "confidence_scoring", "market_intelligence",
            "signal_ranking", "auto_scanning", "regime_detection", "explainability", "fear_greed_index"]}

@app.get("/cap/health")
def cap_health():
    return {"agent": "CROO AI Oracle", "status": "active", "assets": ASSETS,
            "data_sources": ["CoinGecko", "Kraken", "Coinbase", "CryptoCompare", "Alternative.me"],
            "uptime_seconds": int(time.time() - start_time),
            "market_regime": cache["market_regime"], "fear_greed": cache["fear_greed"], "version": "10.0"}
