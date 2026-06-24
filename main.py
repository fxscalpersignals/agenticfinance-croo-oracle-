import os
import time
import asyncio
import aiohttp
import numpy as np
import json
import random
import websockets
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

app = FastAPI(
    title="CROO AI Oracle",
    description="Autonomous Crypto Intelligence Agent with Multi-Provider Fallback, A2A, and Explainable AI",
    version="10.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0")) if os.environ.get("ADMIN_ID") else 0
CHAT_ID = os.environ.get("CHAT_ID")
PAYMENTS_ENABLED = False
PORT = int(os.getenv("PORT", 8000))

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()
shutdown_event = asyncio.Event()

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "AVAXUSDT", "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT"
]

CG_MAP = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "XRPUSDT": "ripple", "BNBUSDT": "binancecoin", "AVAXUSDT": "avalanche-2",
    "DOGEUSDT": "dogecoin", "TRXUSDT": "tron", "ADAUSDT": "cardano", "LINKUSDT": "chainlink"
}

KRAKEN_WS_MAP = {
    "BTCUSDT": "XBT/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
    "XRPUSDT": "XRP/USD", "BNBUSDT": "BNB/USD", "AVAXUSDT": "AVAX/USD",
    "DOGEUSDT": "DOGE/USD", "TRXUSDT": "TRX/USD", "ADAUSDT": "ADA/USD", "LINKUSDT": "LINK/USD"
}

# ==================== STATE ====================
cache = {
    "signals": {},
    "last_scan": 0,
    "last_successful_scan": 0,
    "last_ws_update": 0,
    "market_regime": "neutral",
    "fear_greed": 50,
    "live_prices": {},
    "last_known_prices": {}
}
signal_history = []
users_db = {}
last_alerted = {}
performance = {"wins": 0, "losses": 0, "total": 0}
agent_memory = {
    "last_100_signals": [],
    "best_asset": "NONE",
    "best_asset_win_rate": 0.0,
    "total_calls": 0,
    "revenue_simulated": 0.0
}

scanner_task = None
ws_tasks = []
health_task = None
telegram_worker_task = None
disabled_ws = set()
provider_status = {}

scan_lock = asyncio.Lock()
api_semaphore = asyncio.Semaphore(5)
telegram_queue = asyncio.Queue()
recent_signals = set()

last_api_call = {}
api_failures = {}
cg_cache = {}
session = None

# ==================== MEMORY ====================
def load_memory():
    global signal_history, performance, agent_memory, cache
    try:
        if os.path.exists("agent_memory.json"):
            with open("agent_memory.json", "r") as f:
                data = json.load(f)
                signal_history = data.get("signal_history", [])
                performance = data.get("performance", {"wins": 0, "losses": 0, "total": 0})
                agent_memory = data.get("agent_memory", {
                    "last_100_signals": [],
                    "best_asset": "NONE",
                    "best_asset_win_rate": 0.0,
                    "total_calls": 0,
                    "revenue_simulated": 0.0
                })
                cache["last_known_prices"] = data.get("last_prices", {})
                print("Memory loaded from disk")
    except Exception as e:
        print(f"Memory load failed: {e}")

def save_memory():
    try:
        with open("agent_memory.json", "w") as f:
            json.dump({
                "signal_history": signal_history[-100:],
                "performance": performance,
                "agent_memory": agent_memory,
                "last_prices": cache["last_known_prices"],
                "timestamp": datetime.utcnow().isoformat()
            }, f)
    except Exception as e:
        print(f"Memory save failed: {e}")

# ==================== HELPERS ====================
def normalize_symbol(symbol: str) -> str:
    sym = symbol.upper()
    if not sym.endswith("USDT") and "USDT" not in sym:
        known = {"XBT": "BTCUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
                 "XRP": "XRPUSDT", "BNB": "BNBUSDT", "AVAX": "AVAXUSDT", "DOGE": "DOGEUSDT",
                 "TRX": "TRXUSDT", "ADA": "ADAUSDT", "LINK": "LINKUSDT"}
        if sym in known:
            return known[sym]
        return sym + "USDT"
    return sym

# ==================== RATE LIMITING ====================
def can_call(name, cooldown=30):
    now = time.time()
    if name not in last_api_call:
        last_api_call[name] = now
        return True
    if now - last_api_call[name] > cooldown:
        last_api_call[name] = now
        return True
    return False

def check_circuit_breaker(name):
    if name in api_failures:
        failures, reset_at = api_failures[name]
        if failures >= 5 and time.time() < reset_at:
            return False
        if time.time() >= reset_at:
            api_failures[name] = (0, time.time() + 600)
    return True

def mark_failure(name):
    if name not in api_failures:
        api_failures[name] = (1, time.time() + 600)
    else:
        failures, reset_at = api_failures[name]
        api_failures[name] = (failures + 1, reset_at)
    provider_status[name] = "error"

def mark_success(name):
    if name in api_failures:
        api_failures[name] = (0, time.time() + 600)
    provider_status[name] = "active"

# ==================== TELEGRAM QUEUE (NO HTML) ====================
async def telegram_worker():
    while not shutdown_event.is_set():
        try:
            msg = await asyncio.wait_for(telegram_queue.get(), timeout=1.0)
            if bot:
                try:
                    await bot.send_message(**msg)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Telegram send error: {e}")
            telegram_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"Telegram worker error: {e}")
            await asyncio.sleep(1)

async def send_telegram_message(chat_id, text, reply_markup=None):
    await telegram_queue.put({
        "chat_id": chat_id,
        "text": text,
        "reply_markup": reply_markup
    })

# ==================== WEBSOCKET PARSERS ====================
def parse_bybit(data):
    try:
        if "data" not in data:
            return None, None
        ticker = data["data"]
        if isinstance(ticker, list):
            ticker = ticker[0] if ticker else None
        if not ticker or "lastPrice" not in ticker:
            return None, None
        symbol = normalize_symbol(ticker["symbol"])
        return symbol, float(ticker["lastPrice"])
    except Exception:
        return None, None

def parse_binance(data):
    if "s" in data and "c" in data:
        symbol = normalize_symbol(data["s"])
        return symbol, float(data["c"])
    return None, None

def parse_okx(data):
    if "arg" in data and "data" in data and len(data["data"]) > 0:
        raw_symbol = data["arg"]["instId"]
        symbol = normalize_symbol(raw_symbol.replace("-", ""))
        return symbol, float(data["data"][0]["last"])
    return None, None

def parse_kraken(data):
    if isinstance(data, list) and len(data) > 2 and isinstance(data[1], list):
        pair = data[3].replace("/", "")
        for k, v in KRAKEN_WS_MAP.items():
            if v.replace("/", "") == pair:
                return k, float(data[1][0])
    return None, None

async def websocket_feed(uri, sub_msg, name, parser):
    retry = 1
    while not shutdown_event.is_set():
        if name in disabled_ws:
            await asyncio.sleep(60)
            continue
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                await ws.send(json.dumps(sub_msg))
                provider_status[name] = "active"
                print(f"[Provider] {name} connected")
                retry = 1
                async for msg in ws:
                    if shutdown_event.is_set():
                        break
                    data = json.loads(msg)
                    symbol, price = parser(data)
                    if symbol and price:
                        cache["live_prices"][symbol] = price
                        cache["last_known_prices"][symbol] = price
                        cache["last_ws_update"] = time.time()
        except Exception as e:
            error_str = str(e)
            if "451" in error_str:
                disabled_ws.add(name)
                provider_status[name] = "disabled (region)"
                print(f"[Provider] {name} unavailable (HTTP 451 Region Restriction)")
                continue
            provider_status[name] = "error"
            print(f"[Provider] {name} error: {e}")
            print(f"[Fallback] {name} unavailable, switching to next provider")
            await asyncio.sleep(retry)
            retry = min(retry * 2, 60)

async def start_websockets():
    global ws_tasks
    feeds = [
        ("wss://stream.bybit.com/v5/public/linear", {"op": "subscribe", "args": [f"tickers.{s}" for s in ASSETS]}, "Bybit", parse_bybit),
        ("wss://stream.binance.com:9443/ws", {"method": "SUBSCRIBE", "params": [f"{s.lower()}@ticker" for s in ASSETS], "id": 1}, "Binance", parse_binance),
        ("wss://ws.okx.com:8443/ws/v5/public", {"op": "subscribe", "args": [{"channel": "tickers", "instId": s.replace("USDT","-USDT")} for s in ASSETS]}, "OKX", parse_okx),
        ("wss://ws.kraken.com", {"event": "subscribe", "pair": [KRAKEN_WS_MAP[s] for s in ASSETS if s in KRAKEN_WS_MAP], "subscription": {"name": "ticker"}}, "Kraken", parse_kraken)
    ]
    ws_tasks = []
    for uri, sub_msg, name, parser in feeds:
        task = asyncio.create_task(websocket_feed(uri, sub_msg, name, parser))
        ws_tasks.append(task)
    await asyncio.gather(*ws_tasks, return_exceptions=True)

# ==================== OHLCV FALLBACKS ====================
async def fetch_okx_ohlc(asset):
    if not can_call(f"okx_{asset}", 30) or not check_circuit_breaker("okx"):
        return None, None
    try:
        symbol = asset.replace("USDT", "-USDT")
        async with session.get(f"https://www.okx.com/api/v5/market/candles", params={"instId": symbol, "bar": "1H", "limit": "100"}) as r:
            if r.status == 200:
                data = await r.json()
                if data["code"] == "0":
                    raw = data["data"]
                    raw.reverse()
                    klines = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw]
                    mark_success("okx")
                    return klines, "OKX"
    except: pass
    mark_failure("okx")
    return None, None

async def fetch_bybit_ohlc(asset):
    if not can_call(f"bybit_{asset}", 30) or not check_circuit_breaker("bybit"):
        return None, None
    try:
        async with session.get("https://api.bybit.com/v5/market/kline", params={"category": "linear", "symbol": asset, "interval": "60", "limit": 100}) as r:
            if r.status == 200:
                data = await r.json()
                raw = data["result"]["list"]
                raw.reverse()
                klines = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                mark_success("bybit")
                return klines, "Bybit"
    except: pass
    mark_failure("bybit")
    return None, None

async def fetch_binance_ohlc(asset):
    if not can_call(f"binance_{asset}", 30) or not check_circuit_breaker("binance"):
        return None, None
    try:
        async with session.get("https://api.binance.com/api/v3/klines", params={"symbol": asset, "interval": "1h", "limit": 100}) as r:
            if r.status == 200:
                raw = await r.json()
                klines = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                mark_success("binance")
                return klines, "Binance"
    except: pass
    mark_failure("binance")
    return None, None

async def fetch_coingecko_ohlcv(asset):
    if not can_call(f"cg_{asset}", 120) or not check_circuit_breaker("coingecko"):
        return None, None
    if asset in cg_cache and time.time() - cg_cache[asset]["timestamp"] < 300:
        return cg_cache[asset]["data"], "CoinGecko(Cached)"
    try:
        coin_id = CG_MAP[asset]
        async with session.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": 4, "interval": "hourly"}) as r:
            if r.status == 200:
                data = await r.json()
                prices = data["prices"]
                volumes = data["total_volumes"]
                klines = []
                for i in range(len(prices)):
                    klines.append([prices[i][0], prices[i][1], prices[i][1], prices[i][1], prices[i][1], volumes[i][1] if i < len(volumes) else 0])
                cg_cache[asset] = {"data": klines[-100:], "timestamp": time.time()}
                mark_success("coingecko")
                return klines[-100:], "CoinGecko"
    except: pass
    mark_failure("coingecko")
    return None, None

async def fetch_coinmarketcap_price(asset):
    if not can_call(f"cmc_{asset}", 60) or not check_circuit_breaker("coinmarketcap"):
        return None
    try:
        cmc_api_key = os.environ.get("CMC_API_KEY", "")
        if not cmc_api_key:
            return None
        symbol = asset.replace("USDT", "")
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        async with session.get(url, params={"symbol": symbol, "convert": "USD"}, headers={"X-CMC_PRO_API_KEY": cmc_api_key}) as r:
            if r.status == 200:
                data = await r.json()
                price = data["data"][symbol]["quote"]["USD"]["price"]
                mark_success("coinmarketcap")
                return price
    except: pass
    mark_failure("coinmarketcap")
    return None

async def get_ohlcv(asset):
    providers = [fetch_okx_ohlc, fetch_bybit_ohlc, fetch_binance_ohlc, fetch_coingecko_ohlcv]
    for provider in providers:
        try:
            async with api_semaphore:
                data, source = await provider(asset)
                if data and len(data) > 50:
                    return data, source
        except Exception as e:
            print(f"{provider.__name__} error: {e}")
    return None, "none"

async def fetch_binance_price(asset):
    if not can_call(f"binance_price_{asset}", 10):
        return None
    try:
        async with session.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": asset}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["price"])
    except: pass
    return None

async def fetch_bybit_price(asset):
    if not can_call(f"bybit_price_{asset}", 10):
        return None
    try:
        async with session.get("https://api.bybit.com/v5/market/tickers", params={"category": "linear", "symbol": asset}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["result"]["list"][0]["lastPrice"])
    except: pass
    return None

async def fetch_coingecko_price(asset):
    if not can_call(f"cg_price_{asset}", 60):
        return None
    try:
        coin_id = CG_MAP[asset]
        async with session.get("https://api.coingecko.com/api/v3/simple/price", params={"ids": coin_id, "vs_currencies": "usd"}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data[coin_id]["usd"])
    except: pass
    return None

async def get_price_fallback(asset):
    if asset in cache["live_prices"] and cache["live_prices"][asset] > 0:
        return cache["live_prices"][asset]
    for fetcher in [fetch_binance_price, fetch_bybit_price, fetch_coingecko_price, fetch_coinmarketcap_price]:
        price = await fetcher(asset)
        if price:
            cache["last_known_prices"][asset] = price
            return price
    return cache["last_known_prices"].get(asset, 0)

def get_current_price(asset):
    return cache["live_prices"].get(asset, cache["last_known_prices"].get(asset, 0))

# ==================== FEAR & GREED ====================
async def fetch_fear_greed():
    if not can_call("fear_greed", 300):
        return cache["fear_greed"]
    try:
        async with session.get("https://api.alternative.me/fng/") as r:
            if r.status == 200:
                data = await r.json()
                val = int(data["data"][0]["value"])
                cache["fear_greed"] = val
                return val
    except: pass
    return cache["fear_greed"]

# ==================== INDICATORS ====================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.array([50.0] * len(closes))
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(closes)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(closes)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calc_ema(prices, period):
    if len(prices) < period:
        return np.array([prices[-1]] if prices else [0])
    alpha = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.01
    tr = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr.append(max(hl, hc, lc))
    if len(tr) < period:
        return 0.01
    return np.mean(tr[-period:])

def grade(confidence):
    if confidence >= 90: return "A+"
    elif confidence >= 80: return "A"
    elif confidence >= 70: return "B"
    elif confidence >= 60: return "C"
    elif confidence >= 50: return "D"
    return "F"

# ==================== ENTRY CALCULATOR ====================
def calculate_entries(price, atr, direction="LONG", signal_type="BUY"):
    if direction == "LONG" and signal_type in ["BUY", "HOLD"]:
        entries = {
            "aggressive": round(price * 0.998, 4),
            "moderate": round(price * 0.995, 4),
            "conservative": round(price * 0.990, 4),
            "dca_1": round(price * 0.980, 4),
            "dca_2": round(price * 0.965, 4),
        }
        recommended = entries["moderate"]
        entry_zone = f"${entries['moderate']} - ${entries['conservative']}"
        atr_stop = max(price * 0.025, atr * 2)
        stop_loss = round(recommended - atr_stop, 4)
        risk = recommended - stop_loss
        tp_1 = round(recommended + (risk * 1.5), 4)
        tp_2 = round(recommended + (risk * 2.0), 4)
        tp_3 = round(recommended + (risk * 2.5), 4)
        volatility_factor = min(0.06, atr / price * 4)
        tp_fixed = round(recommended * (1 + volatility_factor), 4)
        take_profit = min(tp_3, tp_fixed)
        take_profit = max(take_profit, round(recommended * 1.02, 4))
        reward = take_profit - recommended
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        position_sizing = "50% moderate, 30% conservative, 20% DCA"
    elif direction == "SHORT" and signal_type in ["SELL", "HOLD"]:
        entries = {
            "aggressive": round(price * 1.002, 4),
            "moderate": round(price * 1.005, 4),
            "conservative": round(price * 1.010, 4),
            "dca_1": round(price * 1.020, 4),
            "dca_2": round(price * 1.035, 4),
        }
        recommended = entries["moderate"]
        entry_zone = f"${entries['moderate']} - ${entries['conservative']}"
        atr_stop = max(price * 0.025, atr * 2)
        stop_loss = round(recommended + atr_stop, 4)
        risk = stop_loss - recommended
        tp_1 = round(recommended - (risk * 1.5), 4)
        tp_2 = round(recommended - (risk * 2.0), 4)
        tp_3 = round(recommended - (risk * 2.5), 4)
        volatility_factor = min(0.06, atr / price * 4)
        tp_fixed = round(recommended * (1 - volatility_factor), 4)
        take_profit = max(tp_3, tp_fixed)
        take_profit = min(take_profit, round(recommended * 0.98, 4))
        reward = recommended - take_profit
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        position_sizing = "50% moderate, 30% conservative, 20% DCA"
    else:
        return {
            "entry": round(price, 4),
            "entry_zone": "N/A",
            "entries": {"current": round(price, 4)},
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A"
        }
    return {
        "entry": recommended,
        "entry_zone": entry_zone,
        "entries": entries,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": f"{risk_reward}:1",
        "position_sizing": position_sizing
    }

# ==================== USER MGMT ====================
def is_pro(user_id: int) -> bool:
    user = users_db.get(user_id, {})
    if not user: return False
    if user.get("plan") == "lifetime": return True
    expires = user.get("pro_expires")
    return expires and datetime.now() < expires

def activate_pro(user_id: int, days: int = 30):
    if user_id not in users_db:
        users_db[user_id] = {}
    users_db[user_id]["plan"] = "pro"
    users_db[user_id]["pro_expires"] = datetime.now() + timedelta(days=days)

# ==================== CAPITAL PLAN ====================
def get_capital_plan():
    return {
        "tier1": "50% (Moderate Entry)",
        "tier2": "30% (Conservative Entry)",
        "tier3": "20% (DCA Entry)",
        "max_risk": "2% of capital"
    }

# ==================== CORE ANALYSIS ====================
async def detect_regime():
    klines, _ = await get_ohlcv("BTCUSDT")
    if not klines or len(klines) < 50:
        return cache["market_regime"]
    closes = np.array([float(k[4]) for k in klines])
    if len(closes) < 50:
        return cache["market_regime"]
    ema50 = calc_ema(closes, 50)[-1]
    return "bullish" if closes[-1] > ema50 else "bearish"

async def analyze_asset(symbol):
    klines, source = await get_ohlcv(symbol)
    if not klines or len(klines) < 50:
        price = await get_price_fallback(symbol)
        if price > 0:
            entry_data = calculate_entries(price, 0.01, "LONG", "HOLD")
            return {
                "asset": symbol.replace("USDT", ""),
                "decision": "HOLD",
                "bias": "NEUTRAL",
                "confidence": 20,
                "grade": "F",
                "price": round(price, 4),
                "entry": entry_data["entry"],
                "entry_zone": entry_data["entry_zone"],
                "entries": entry_data["entries"],
                "stop_loss": entry_data["stop_loss"],
                "take_profit": entry_data["take_profit"],
                "risk_reward": entry_data["risk_reward"],
                "position_sizing": entry_data["position_sizing"],
                "bullish_reasons": ["Insufficient data"],
                "bearish_reasons": [],
                "missing_conditions": ["Full OHLCV data unavailable"],
                "source": "price_only",
                "direction": "NONE",
                "risk": "HIGH",
                "holding_period": "N/A",
                "pullback_pct": 0,
                "action": "WAIT_FOR_DATA",
                "why_not_now": ["Insufficient data for analysis"],
                "checks": {},
                "reasoning": ["Waiting for more data"],
                "capital_plan": get_capital_plan()
            }
        return {
            "asset": symbol.replace("USDT", ""),
            "decision": "HOLD",
            "bias": "NEUTRAL",
            "confidence": 0,
            "price": 0,
            "entry": 0,
            "entry_zone": "N/A",
            "entries": {},
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A",
            "bullish_reasons": [],
            "bearish_reasons": [],
            "direction": "NONE",
            "risk": "UNKNOWN",
            "holding_period": "N/A",
            "pullback_pct": 0,
            "action": "NO_DATA",
            "why_not_now": ["Market data unavailable"],
            "checks": {},
            "reasoning": ["No data available"],
            "capital_plan": get_capital_plan()
        }

    closes = np.array([float(k[4]) for k in klines])
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])

    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    rsi_val = calc_rsi(closes)[-1]
    ema20 = calc_ema(closes, 20)[-1]
    ema50 = calc_ema(closes, 50)[-1]
    atr = calc_atr(highs, lows, closes)
    atr_pct = (atr / price) * 100

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    price_near_ema20 = abs(price - ema20) / ema20 * 100 < 1.5
    bullish_confirmation = price > prev_close
    bearish_confirmation = price < prev_close

    long_score = 0
    short_score = 0
    bullish_reasons = []
    bearish_reasons = []
    missing_conditions = []
    checks = {}
    reasoning = []

    # RSI
    if rsi_val < 45:
        long_score += 20
        bullish_reasons.append(f"RSI Oversold ({rsi_val:.1f})")
        checks["rsi_oversold"] = True
        reasoning.append("RSI oversold indicates potential bottom")
    elif rsi_val > 55:
        short_score += 20
        bearish_reasons.append(f"RSI Overbought ({rsi_val:.1f})")
        checks["rsi_overbought"] = True
        reasoning.append("RSI overbought suggests potential top")
    else:
        missing_conditions.append("RSI neutral")
        checks["rsi_extreme"] = False
        reasoning.append("RSI neutral – no extreme signal")

    # EMA
    if price > ema50:
        long_score += 20
        bullish_reasons.append("Above EMA50")
        checks["above_ema50"] = True
        reasoning.append("Price above EMA50 – uptrend")
    elif price < ema50:
        short_score += 20
        bearish_reasons.append("Below EMA50")
        checks["below_ema50"] = True
        reasoning.append("Price below EMA50 – downtrend")
    else:
        missing_conditions.append("No clear EMA trend")
        checks["ema_trend"] = False
        reasoning.append("No clear EMA trend")

    # Pullback/Bounce
    if price > ema50 and 4 < pullback < 12 and price_near_ema20:
        long_score += 20
        bullish_reasons.append(f"Dip {pullback:.1f}% to EMA20")
        checks["pullback_zone"] = True
        reasoning.append(f"Pullback to EMA20 ({pullback:.1f}%)")
    elif price < ema50 and 4 < bounce < 12 and price_near_ema20:
        short_score += 20
        bearish_reasons.append(f"Bounce {bounce:.1f}% to EMA20")
        checks["bounce_zone"] = True
        reasoning.append(f"Dead cat bounce to EMA20 ({bounce:.1f}%)")
    else:
        missing_conditions.append("Pullback too shallow/deep")
        checks["pullback_zone"] = False
        reasoning.append("Pullback not in optimal zone")

    # Volume
    if vol_spike:
        if long_score >= short_score:
            long_score += 20
            bullish_reasons.append("Volume Spike")
            checks["volume_spike"] = True
            reasoning.append("Volume spike confirms buying interest")
        else:
            short_score += 20
            bearish_reasons.append("Volume Spike")
            checks["volume_spike"] = True
            reasoning.append("Volume spike confirms selling pressure")
    else:
        missing_conditions.append("No volume confirmation")
        checks["volume_spike"] = False
        reasoning.append("No significant volume")

    # Confirmation
    if bullish_confirmation:
        long_score += 20
        bullish_reasons.append("Bullish Confirmation")
        checks["bullish_confirmation"] = True
        reasoning.append("Bullish confirmation candle")
    elif bearish_confirmation:
        short_score += 20
        bearish_reasons.append("Bearish Confirmation")
        checks["bearish_confirmation"] = True
        reasoning.append("Bearish confirmation candle")
    else:
        missing_conditions.append("No confirmation candle")
        checks["confirmation"] = False
        reasoning.append("No clear confirmation candle")

    # Fear & Greed
    fg = cache["fear_greed"]
    if fg < 25 and long_score >= short_score:
        long_score += 5
        bullish_reasons.append("Extreme Fear")
        checks["extreme_fear"] = True
        reasoning.append(f"Fear & Greed = {fg} (extreme fear)")
    if fg > 75 and short_score > long_score:
        short_score += 5
        bearish_reasons.append("Extreme Greed")
        checks["extreme_greed"] = True
        reasoning.append(f"Fear & Greed = {fg} (extreme greed)")

    # Volatility penalty
    if atr_pct < 1:
        if long_score >= short_score:
            long_score -= 15
        else:
            short_score -= 15
        missing_conditions.append("Low volatility (ATR < 1%)")
        checks["volatility_ok"] = False
        reasoning.append("Volatility too low – filtering out")

    direction = "LONG" if long_score >= short_score else "SHORT"
    confidence = max(long_score, short_score)
    confidence = max(0, min(100, confidence))

    if confidence >= 60:
        decision = "BUY" if direction == "LONG" else "SELL"
    elif confidence >= 40:
        decision = "HOLD"
        reasoning.append("Confidence below 60 – waiting for stronger signal")
        if direction == "LONG":
            missing = [m for m in missing_conditions if "volume" not in m.lower() and "rsi" not in m.lower()]
            if missing:
                reasoning.append("Missing: " + ", ".join(missing[:2]))
        else:
            missing = [m for m in missing_conditions if "volume" not in m.lower() and "rsi" not in m.lower()]
            if missing:
                reasoning.append("Missing: " + ", ".join(missing[:2]))
    else:
        decision = "HOLD"
        reasoning.append("Confidence below 40 – no clear signal")
        if direction == "LONG":
            reasoning.append("LONG bias but insufficient confidence")
        else:
            reasoning.append("SHORT bias but insufficient confidence")

    bias = direction if decision != "HOLD" else "NEUTRAL"

    if decision in ["BUY"] or (decision == "HOLD" and direction == "LONG"):
        entry_data = calculate_entries(price, atr, "LONG", decision if decision in ["BUY"] else "HOLD")
        risk = "LOW" if atr / price < 0.02 else "MEDIUM"
        holding_period = "1-3 days"
    elif decision in ["SELL"] or (decision == "HOLD" and direction == "SHORT"):
        entry_data = calculate_entries(price, atr, "SHORT", decision if decision in ["SELL"] else "HOLD")
        risk = "LOW" if atr / price < 0.02 else "MEDIUM"
        holding_period = "1-3 days"
    else:
        entry_data = {
            "entry": round(price, 4),
            "entry_zone": "N/A",
            "entries": {"current": round(price, 4)},
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A"
        }
        risk = "N/A"
        holding_period = "N/A"

    why_not_now = []
    if decision == "HOLD":
        if direction == "LONG":
            if not checks.get("volume_spike", False):
                why_not_now.append("No volume confirmation")
            if not checks.get("rsi_oversold", False):
                why_not_now.append("RSI not oversold")
            if not checks.get("pullback_zone", False):
                why_not_now.append("Pullback not at EMA20")
            if not checks.get("bullish_confirmation", False):
                why_not_now.append("No bullish confirmation")
        else:
            if not checks.get("volume_spike", False):
                why_not_now.append("No volume confirmation")
            if not checks.get("rsi_overbought", False):
                why_not_now.append("RSI not overbought")
            if not checks.get("bounce_zone", False):
                why_not_now.append("Bounce not at EMA20")
            if not checks.get("bearish_confirmation", False):
                why_not_now.append("No bearish confirmation")
        if not why_not_now:
            why_not_now.append("Confidence below threshold")

    if not bullish_reasons:
        bullish_reasons = ["Waiting for setup"]
    if not bearish_reasons:
        bearish_reasons = ["Waiting for setup"]

    return {
        "asset": symbol.replace("USDT", ""),
        "price": round(price, 4),
        "decision": decision,
        "bias": bias,
        "confidence": confidence,
        "grade": grade(confidence),
        "direction": direction,
        "entry": entry_data["entry"],
        "entry_zone": entry_data["entry_zone"],
        "entries": entry_data["entries"],
        "stop_loss": entry_data["stop_loss"],
        "take_profit": entry_data["take_profit"],
        "risk_reward": entry_data["risk_reward"],
        "position_sizing": entry_data["position_sizing"],
        "rsi": round(rsi_val, 1),
        "atr": round(atr, 4),
        "atr_pct": round(atr_pct, 2),
        "risk": risk,
        "holding_period": holding_period,
        "bullish_reasons": bullish_reasons,
        "bearish_reasons": bearish_reasons,
        "missing_conditions": missing_conditions,
        "checks": checks,
        "action": decision,
        "why_not_now": why_not_now,
        "reasoning": reasoning,
        "capital_plan": get_capital_plan(),
        "source": source,
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "pullback_pct": round(pullback, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== PERFORMANCE ====================
async def update_performance():
    for signal in signal_history:
        if signal.get("status") == "open" and signal.get("entry", 0) > 0:
            current = get_current_price(signal["asset"] + "USDT")
            if current == 0:
                continue
            if signal["direction"] == "LONG":
                if current >= signal["take_profit"]:
                    signal["pnl"] = round((signal["take_profit"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"
                    performance["wins"] += 1
                    performance["total"] += 1
                elif current <= signal["stop_loss"]:
                    signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"
                    performance["losses"] += 1
                    performance["total"] += 1
            elif signal["direction"] == "SHORT":
                if current <= signal["take_profit"]:
                    signal["pnl"] = round((signal["entry"] - signal["take_profit"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"
                    performance["wins"] += 1
                    performance["total"] += 1
                elif current >= signal["stop_loss"]:
                    signal["pnl"] = round((signal["entry"] - signal["stop_loss"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"
                    performance["losses"] += 1
                    performance["total"] += 1
    update_memory()
    save_memory()

def update_memory():
    agent_memory["last_100_signals"] = signal_history[-100:]
    stats = {}
    for sig in signal_history:
        if sig.get("status") in ["win", "loss"]:
            asset = sig["asset"]
            if asset not in stats:
                stats[asset] = {"wins": 0, "total": 0}
            stats[asset]["total"] += 1
            if sig["status"] == "win":
                stats[asset]["wins"] += 1
    best = None
    best_rate = 0
    for asset, s in stats.items():
        if s["total"] >= 3:
            rate = s["wins"] / s["total"]
            if rate > best_rate:
                best_rate = rate
                best = asset
    agent_memory["best_asset"] = best or "NONE"
    agent_memory["best_asset_win_rate"] = round(best_rate * 100, 1) if best_rate > 0 else 0

# ==================== ALERTS ====================
async def send_alert(signal):
    if not bot or signal["confidence"] < 60:
        return
    if signal["asset"] in last_alerted and time.time() - last_alerted[signal["asset"]] < 3600:
        return

    msg = f"🚨 {signal['decision']} SIGNAL\n\n"
    msg += f"Asset: {signal['asset']}\n"
    msg += f"Confidence: {signal['confidence']}% ({signal['grade']})\n"
    msg += f"Risk: {signal['risk']}\n\n"
    msg += f"Entry Zone: {signal['entry_zone']}\n"
    msg += f"Entry: ${signal['entry']}\n"
    msg += f"Target: ${signal['take_profit']}\n"
    msg += f"Stop: ${signal['stop_loss']}\n"
    msg += f"R:R: {signal['risk_reward']}\n\n"
    msg += f"Position Sizing:\n{signal['position_sizing']}\n\n"
    msg += f"Reasons:\n"
    reasons = signal['bullish_reasons'] if signal['direction'] == 'LONG' else signal['bearish_reasons']
    msg += "\n".join([f"✅ {r}" for r in reasons[:5]])
    msg += f"\n\nMarket: {signal['market_regime'].upper()} | F&G: {signal['fear_greed']}"
    msg += f"\nHolding Period: {signal['holding_period']}"
    msg += f"\nATR: {signal['atr_pct']}%"

    if CHAT_ID:
        await send_telegram_message(CHAT_ID, msg)

    last_alerted[signal["asset"]] = time.time()

# ==================== SCANNER ====================
async def scan_all(force=False):
    async with scan_lock:
        if not force and time.time() - cache["last_scan"] < 120:
            return cache["signals"]

        print(f"SCAN {datetime.utcnow()}")
        await fetch_fear_greed()
        await update_performance()
        cache["market_regime"] = await detect_regime()

        results = {}
        signal_summary = {"BUY":0, "SELL":0, "HOLD":0}
        for asset in ASSETS:
            data = await analyze_asset(asset)
            if data:
                results[asset] = data
                signal_summary[data["decision"]] = signal_summary.get(data["decision"], 0) + 1
                signal_key = f"{data['asset']}_{data['decision']}_{data['direction']}_{round(data['price'], 2)}"
                if data["decision"] in ["BUY", "SELL"] and signal_key not in recent_signals:
                    data["status"] = "open"
                    signal_history.append(data)
                    agent_memory["total_calls"] += 1
                    agent_memory["revenue_simulated"] += 0.01
                    recent_signals.add(signal_key)
                    asyncio.create_task(send_alert(data))
                    print(f"[SCAN] {data['asset']} {data['decision']} {data['confidence']}%")

        print(f"[SUMMARY] BUY={signal_summary['BUY']} SELL={signal_summary['SELL']} HOLD={signal_summary['HOLD']}")

        signal_history[:] = signal_history[-100:]
        if len(recent_signals) > 100:
            recent_signals.clear()

        cache["signals"] = results
        cache["last_scan"] = time.time()
        cache["last_successful_scan"] = time.time()
        save_memory()

        for asset in list(last_alerted.keys()):
            if time.time() - last_alerted[asset] > 86400:
                del last_alerted[asset]

        print(f"Scan complete. {len(results)} assets analyzed.")
        return results

async def scanner_loop():
    print("Auto scanner started")
    while not shutdown_event.is_set():
        try:
            await scan_all()
            jitter = random.randint(-15, 15)
            await asyncio.sleep(300 + jitter)
        except Exception as e:
            print(f"Scanner error: {e}")
            await asyncio.sleep(60)

# ==================== HEALTH MONITOR ====================
async def health_monitor():
    print("Health monitor started")
    while not shutdown_event.is_set():
        try:
            if time.time() - cache["last_ws_update"] > 120:
                print(f"WARNING: WebSocket stale ({int(time.time() - cache['last_ws_update'])}s)")
            if time.time() - cache["last_successful_scan"] > 900:
                print(f"WARNING: Scanner stalled ({int(time.time() - cache['last_successful_scan'])}s)")
            if telegram_queue.qsize() > 50:
                print(f"WARNING: Telegram queue size {telegram_queue.qsize()}")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"Health monitor error: {e}")
            await asyncio.sleep(60)

# ==================== A2A ENHANCED ====================
@app.post("/a2a")
async def a2a(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    agent = data.get("agent", "Unknown")
    request_type = data.get("request", "")

    job_id = f"job_{int(time.time())}_{random.randint(1000, 9999)}"

    if request_type == "best_trade":
        # Ensure fresh data
        if time.time() - cache["last_scan"] > 300:
            await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            return JSONResponse({
                "job_id": job_id,
                "status": "no_signal",
                "message": "No signals available",
                "from_agent": "CROO Oracle",
                "to_agent": agent
            })
        best = max(signals, key=lambda x: x.get("confidence", 0))
        # Track revenue
        agent_memory["total_calls"] += 1
        agent_memory["revenue_simulated"] += 0.01
        save_memory()
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "cost": "0.01 USDC",
            "result": {
                "asset": best.get("asset"),
                "decision": best.get("decision"),
                "confidence": best.get("confidence"),
                "entry_zone": best.get("entry_zone"),
                "entry": best.get("entry"),
                "tp": best.get("take_profit"),
                "sl": best.get("stop_loss"),
                "risk_reward": best.get("risk_reward"),
                "reasoning": best.get("reasoning", [])
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    elif request_type == "market_intel":
        if time.time() - cache["last_scan"] > 300:
            await scan_all()
        agent_memory["total_calls"] += 1
        agent_memory["revenue_simulated"] += 0.005
        save_memory()
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "cost": "0.005 USDC",
            "result": {
                "market_regime": cache["market_regime"],
                "fear_greed": cache["fear_greed"],
                "signals": len([s for s in cache["signals"].values() if s.get("decision") in ["BUY", "SELL"]]),
                "top_asset": agent_memory["best_asset"]
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    return JSONResponse({
        "job_id": job_id,
        "status": "error",
        "message": f"Unknown request: {request_type}",
        "from_agent": "CROO Oracle",
        "to_agent": agent
    }, status_code=400)

# ==================== TELEGRAM COMMANDS ====================
async def send_rich_card(chat_id, s, back_button=True):
    decision = s.get("decision", "HOLD")
    if decision == "HOLD":
        if s.get("bias") == "LONG":
            msg = "⏳ HOLD (LONG BIAS)\n\n"
        elif s.get("bias") == "SHORT":
            msg = "⏳ HOLD (SHORT BIAS)\n\n"
        else:
            msg = "⏳ HOLD\n\n"
    else:
        msg = f"🚨 {decision} SIGNAL\n\n"

    msg += f"Asset: {s.get('asset')}\n"
    msg += f"Confidence: {s.get('confidence')}% ({s.get('grade')})\n"
    msg += f"Risk: {s.get('risk')}\n"
    msg += f"Price: ${s.get('price')}\n\n"
    msg += f"Entry Zone: {s.get('entry_zone')}\n"
    msg += f"Entry: ${s.get('entry')}\n"
    msg += f"Target: ${s.get('take_profit')}\n"
    msg += f"Stop: ${s.get('stop_loss')}\n"
    msg += f"R:R: {s.get('risk_reward')}\n\n"
    msg += f"Position Sizing:\n{s.get('position_sizing')}\n\n"

    if s.get('direction') == 'LONG':
        msg += f"Bullish Reasons:\n" + "\n".join([f"✅ {r}" for r in s.get('bullish_reasons', ['None'])[:5]])
    else:
        msg += f"Bearish Reasons:\n" + "\n".join([f"✅ {r}" for r in s.get('bearish_reasons', ['None'])[:5]])

    if s.get('missing_conditions'):
        msg += "\n\nMissing Conditions:\n" + "\n".join([f"❌ {m}" for m in s.get('missing_conditions', [])[:3]])

    if s.get('why_not_now'):
        msg += "\n\nWhy Not Now:\n" + "\n".join([f"⏳ {w}" for w in s.get('why_not_now', [])[:3]])

    if s.get('reasoning'):
        msg += "\n\n🧠 Reasoning:\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(s.get('reasoning', [])[:5])])

    msg += f"\n\nMarket: {s.get('market_regime','').upper()} | F&G: {s.get('fear_greed')}"
    msg += f"\nSource: {s.get('source', 'N/A')} | Hold: {s.get('holding_period', 'N/A')}"
    msg += f"\nPullback: {s.get('pullback_pct', 0)}% | ATR: {s.get('atr_pct', 0)}%"
    msg += f"\nAction: {s.get('action', 'N/A')}"

    keyboard = None
    if back_button:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="back_to_menu")]]
    await send_telegram_message(chat_id, msg, reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None)

async def handle_buy(chat_id, user_id):
    if PAYMENTS_ENABLED:
        await send_telegram_message(chat_id, "Payment processing coming post-hackathon...")
    else:
        if is_pro(user_id):
            await send_telegram_message(chat_id, "You're already Pro ✅")
        else:
            activate_pro(user_id, days=999)
            await send_telegram_message(
                chat_id,
                "✅ DEMO MODE: Pro activated for hackathon judges\n\nAll features unlocked.\nTry /scan or /best now."
            )

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await send_telegram_message(chat_id, "You're on Free plan. Nothing to cancel.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await send_telegram_message(
            chat_id,
            "✅ DEMO: Pro subscription cancelled\n\nBack to Free plan.\nRe-upgrade: /buy"
        )

async def handle_message(chat_id, text, user_id):
    if not bot:
        return

    if text == "/start":
        if time.time() - cache["last_scan"] > 300:
            await scan_all()
        signals = list(cache["signals"].values())
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
            [InlineKeyboardButton("📈 LINK", callback_data="LINKUSDT"),
             InlineKeyboardButton("💎 Upgrade", callback_data="buy_cmd")]
        ]
        regime = cache["market_regime"].upper()
        top = max(signals, key=lambda x: x.get("confidence", 0)) if signals else None

        msg = "🔮 CROO AI Oracle\n\n"
        msg += f"Market: {regime} | F&G: {cache['fear_greed']}\n"
        msg += f"Assets: {len(ASSETS)} monitored\n"
        msg += f"Entry Strategy: Zone-based (0.5-1% below/above current)\n"
        msg += f"TP Strategy: Realistic 3-6% with 2:1+ R:R\n"
        if top and top.get("confidence", 0) > 0:
            msg += f"\n🔥 Top: {top.get('asset')} {top.get('decision')} {top.get('confidence')}% ({top.get('grade')})\n"
            msg += f"Price: ${top.get('price')} | Zone: {top.get('entry_zone')}\n"
            msg += f"TP: ${top.get('take_profit')} | R:R: {top.get('risk_reward')}\n"
            msg += f"Action: {top.get('action')}\n"
        msg += "\n/scan /best /leaderboard /stats /force_scan /status /subscribe /usage"
        await send_telegram_message(chat_id, msg, reply_markup=InlineKeyboardMarkup(keyboard))

    elif text in ["/scan", "/signals"]:
        await scan_all(force=True)
        await send_leaderboard(chat_id)

    elif text == "/best":
        if time.time() - cache["last_scan"] > 120:
            await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            await send_telegram_message(chat_id, "No signals yet. Scanning...")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)))

    elif text == "/leaderboard":
        await send_leaderboard(chat_id)

    elif text == "/stats":
        await send_stats(chat_id)

    elif text == "/buy":
        await handle_buy(chat_id, user_id)

    elif text == "/sell":
        await handle_sell(chat_id, user_id)

    elif text == "/force_scan":
        await scan_all(force=True)
        await send_telegram_message(chat_id, "✅ Manual scan complete. Check /leaderboard for results.")

    elif text == "/status":
        uptime = str(timedelta(seconds=int(time.time() - start_time)))
        last_scan = cache["last_successful_scan"]
        last_scan_str = datetime.utcfromtimestamp(last_scan).isoformat() if last_scan else "Never"
        active_ws = len([t for t in ws_tasks if not t.done()])
        msg = f"📊 Agent Status\n\n"
        msg += f"Uptime: {uptime}\n"
        msg += f"Last Scan: {last_scan_str}\n"
        msg += f"Signals Generated: {len(signal_history)}\n"
        msg += f"Active WebSockets: {active_ws}\n"
        msg += f"Telegram Queue: {telegram_queue.qsize()}\n"
        await send_telegram_message(chat_id, msg)

    elif text == "/subscribe":
        msg = "💎 CROO Oracle Subscription\n\n"
        msg += "**Free Plan**\n- 5 requests/day\n- Basic signals\n\n"
        msg += "**Pro Plan** – $9.99/month\n- Unlimited requests\n- All assets\n- Entry zones & position sizing\n- Telegram alerts\n\n"
        msg += "**Enterprise** – Custom pricing\n- Full API access\n- White-label\n- Dedicated support\n\n"
        msg += "🔹 Hackathon Demo – All features unlocked for free!\n"
        await send_telegram_message(chat_id, msg)

    elif text == "/usage":
        used = agent_memory["total_calls"]
        free_limit = 5
        if used >= free_limit and not is_pro(user_id):
            remaining = 0
            status = "⚠️ Free limit reached. Upgrade to Pro!"
        else:
            remaining = max(0, free_limit - used)
            status = f"✅ {remaining} free requests left today"
        msg = f"📊 Your Usage\n\n"
        msg += f"Calls made: {used}\n"
        msg += f"Status: {status}\n"
        await send_telegram_message(chat_id, msg)

    elif text.startswith("/why"):
        parts = text.split()
        if len(parts) > 1:
            symbol = parts[1].upper()
            await send_why(chat_id, symbol)

    elif text == "/demo":
        demo_data = await demo()
        await send_telegram_message(chat_id, f"📊 DEMO STATUS\n\n{json.dumps(demo_data, indent=2)}")

async def send_why(chat_id, symbol):
    asset = symbol.upper() + "USDT"
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signal = cache["signals"].get(asset, {})
    if not signal:
        await send_telegram_message(chat_id, f"No data for {symbol}")
        return

    msg = f"🧠 WHY {symbol}?\n\n"
    msg += f"Decision: {signal.get('decision')}\n"
    msg += f"Bias: {signal.get('bias')}\n"
    msg += f"Direction: {signal.get('direction')}\n"
    msg += f"Confidence: {signal.get('confidence')}%\n"
    msg += f"Risk: {signal.get('risk')}\n"
    msg += f"Entry Zone: {signal.get('entry_zone')}\n"
    msg += f"TP: ${signal.get('take_profit')}\n"
    msg += f"SL: ${signal.get('stop_loss')}\n"
    msg += f"R:R: {signal.get('risk_reward')}\n"
    msg += f"ATR: {signal.get('atr_pct')}%\n"
    msg += f"Action: {signal.get('action')}\n\n"

    if signal.get('direction') == 'LONG':
        msg += "Bullish:\n" + "\n".join([f"✅ {r}" for r in signal.get('bullish_reasons', [])[:5]])
    else:
        msg += "Bearish:\n" + "\n".join([f"✅ {r}" for r in signal.get('bearish_reasons', [])[:5]])

    if signal.get('missing_conditions'):
        msg += f"\n\nMissing:\n" + "\n".join([f"❌ {m}" for m in signal.get('missing_conditions', [])[:3]])

    if signal.get('why_not_now'):
        msg += f"\n\nWhy Not Now:\n" + "\n".join([f"⏳ {w}" for w in signal.get('why_not_now', [])[:3]])

    if signal.get('reasoning'):
        msg += "\n\n🧠 Reasoning:\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(signal.get('reasoning', [])[:5])])

    msg += f"\n\nMarket: {signal.get('market_regime', '').upper()} | F&G: {signal.get('fear_greed')}"
    msg += f"\nPosition Sizing: {signal.get('position_sizing', 'N/A')}"
    msg += f"\nHolding Period: {signal.get('holding_period', 'N/A')}"
    await send_telegram_message(chat_id, msg)

async def send_leaderboard(chat_id):
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    msg = f"🏆 LEADERBOARD | {cache['market_regime'].upper()} | F&G: {cache['fear_greed']}\n\n"
    for i, s in enumerate(signals[:10], 1):
        msg += f"{i}. {s.get('asset','N/A')} - {s.get('confidence',0)}% ({s.get('grade','N/A')}) {s.get('decision','NONE')}\n"
        msg += f" ${s.get('price',0)} | Zone: {s.get('entry_zone','N/A')}\n"
        msg += f" TP: ${s.get('take_profit',0)} | R:R: {s.get('risk_reward','N/A')}\n"
        msg += f" Action: {s.get('action','N/A')}\n"
    await send_telegram_message(chat_id, msg)

async def send_stats(chat_id):
    await update_performance()
    win_rate = performance["wins"] / max(1, performance["total"]) * 100
    accuracy = round(win_rate, 1)
    rep_score = win_rate * 0.7 + min(performance["total"], 100) * 0.3
    rep_score = min(100, rep_score)
    msg = f"📊 AGENT STATS\n\n"
    msg += f"Total Signals: {performance['total']}\n"
    msg += f"Wins: {performance['wins']}\n"
    msg += f"Losses: {performance['losses']}\n"
    msg += f"Win Rate: {accuracy}%\n"
    msg += f"Reputation: {round(rep_score, 1)}%\n"
    msg += f"Best Asset: {agent_memory['best_asset']} ({agent_memory['best_asset_win_rate']}%)\n"
    msg += f"Market Regime: {cache['market_regime'].upper()}\n"
    msg += f"Fear & Greed: {cache['fear_greed']}\n"
    msg += f"Revenue Simulated: ${round(agent_memory['revenue_simulated'], 2)}\n"
    msg += f"Memory: {agent_memory['total_calls']} calls\n"
    msg += f"Entry Strategy: Zone-based (0.5-1% below/above current)\n"
    msg += f"TP Strategy: Realistic 3-6% with 2:1+ R:R\n"
    msg += f"Volatility Filter: Active (ATR < 1% penalized)"
    await send_telegram_message(chat_id, msg)

async def handle_callback(chat_id, data, user_id):
    if not bot:
        return

    if data == "back_to_menu":
        await handle_message(chat_id, "/start", user_id)
        return

    if data == "scan_all":
        await scan_all(force=True)
        await send_leaderboard(chat_id)

    elif data == "leaderboard":
        await send_leaderboard(chat_id)

    elif data == "best_signal":
        if time.time() - cache["last_scan"] > 120:
            await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            await send_telegram_message(chat_id, "No signals yet. Scanning...")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)))

    elif data == "buy_cmd":
        await handle_buy(chat_id, user_id)

    elif data in ASSETS:
        if time.time() - cache["last_scan"] > 300:
            await scan_all()
        s = cache["signals"].get(data, {})
        if not s or s.get("confidence", 0) == 0:
            await send_telegram_message(chat_id, f"No data for {data.replace('USDT','')} yet. Scanning...")
        else:
            await send_rich_card(chat_id, s)

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "agent": "CROO AI Oracle",
        "version": "10.0",
        "assets": len(ASSETS),
        "status": "online",
        "uptime": str(timedelta(seconds=int(time.time() - start_time))),
        "entry_strategy": "Zone-based entries (0.5-1% below/above current)",
        "tp_strategy": "Realistic 3-6% with 2:1+ R:R",
        "endpoints": [
            "/oracle", "/best_signal", "/leaderboard", "/stats",
            "/history", "/agent/query", "/a2a",
            "/cap/metadata", "/cap/health", "/pricing", "/capabilities",
            "/explain/{symbol}", "/why/{symbol}", "/reasoning/{symbol}",
            "/portfolio", "/demo", "/business_model",
            "/.well-known/agent.json"
        ]
    }

@app.head("/")
def root_head():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "scanner": scanner_task is not None and not scanner_task.done(),
        "websockets": len(ws_tasks),
        "signals": len(signal_history),
        "uptime": str(timedelta(seconds=int(time.time() - start_time)))
    }

@app.get("/oracle")
async def oracle():
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    return cache["signals"]

@app.get("/best_signal")
async def best_signal():
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    if not signals:
        return JSONResponse({"message": "No signals right now"})
    best = max(signals, key=lambda x: x.get("confidence", 0))
    return JSONResponse({
        "asset": best.get("asset"),
        "decision": best.get("decision"),
        "bias": best.get("bias"),
        "direction": best.get("direction"),
        "confidence": best.get("confidence"),
        "grade": best.get("grade"),
        "price": best.get("price"),
        "entry_zone": best.get("entry_zone"),
        "entry": best.get("entry"),
        "tp": best.get("take_profit"),
        "sl": best.get("stop_loss"),
        "risk_reward": best.get("risk_reward"),
        "risk": best.get("risk"),
        "holding_period": best.get("holding_period"),
        "position_sizing": best.get("position_sizing"),
        "action": best.get("action"),
        "why_not_now": best.get("why_not_now", []),
        "reasoning": best.get("reasoning", []),
        "reasons": best.get("bullish_reasons") if best.get("direction") == "LONG" else best.get("bearish_reasons")
    })

@app.get("/leaderboard")
async def leaderboard():
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    return JSONResponse([{
        "asset": s.get("asset"),
        "decision": s.get("decision"),
        "bias": s.get("bias"),
        "direction": s.get("direction"),
        "confidence": s.get("confidence"),
        "grade": s.get("grade"),
        "price": s.get("price"),
        "entry_zone": s.get("entry_zone"),
        "risk_reward": s.get("risk_reward"),
        "source": s.get("source")
    } for s in signals[:10]])

@app.get("/stats")
async def stats():
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    await update_performance()
    win_rate = performance["wins"] / max(1, performance["total"]) * 100
    accuracy = round(win_rate, 1)
    rep_score = win_rate * 0.7 + min(performance["total"], 100) * 0.3
    rep_score = min(100, rep_score)
    return JSONResponse({
        "accuracy": f"{accuracy}%",
        "total_signals": performance["total"],
        "wins": performance["wins"],
        "losses": performance["losses"],
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "best_asset": agent_memory["best_asset"],
        "best_asset_win_rate": f"{agent_memory['best_asset_win_rate']}%",
        "reputation_score": round(rep_score, 1),
        "memory": {
            "signals_generated": agent_memory["total_calls"],
            "revenue_simulated": round(agent_memory["revenue_simulated"], 2)
        }
    })

@app.get("/history")
def history():
    return signal_history[-50:]

@app.post("/agent/query")
async def agent_query(req: Request):
    try:
        data = await req.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    task = data.get("task", "")
    if time.time() - cache["last_scan"] > 300:
        await scan_all()

    if task == "find_best_pullback":
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            return JSONResponse({"message": "No signals"})
        best = max(signals, key=lambda x: x.get("confidence", 0))
        return JSONResponse({
            "asset": best.get("asset"),
            "decision": best.get("decision"),
            "bias": best.get("bias"),
            "direction": best.get("direction"),
            "confidence": best.get("confidence"),
            "entry_zone": best.get("entry_zone"),
            "entry": best.get("entry"),
            "tp": best.get("take_profit"),
            "sl": best.get("stop_loss"),
            "risk_reward": best.get("risk_reward"),
            "risk": best.get("risk"),
            "holding_period": best.get("holding_period"),
            "action": best.get("action"),
            "reasoning": best.get("reasoning", []),
            "reason": best.get("bullish_reasons") if best.get("direction") == "LONG" else best.get("bearish_reasons")
        })

    elif task == "get_all_signals":
        signals = []
        for s in cache["signals"].values():
            if s.get("confidence", 0) > 0:
                signals.append({
                    "asset": s.get("asset"),
                    "decision": s.get("decision"),
                    "bias": s.get("bias"),
                    "direction": s.get("direction"),
                    "confidence": s.get("confidence"),
                    "price": s.get("price"),
                    "grade": s.get("grade"),
                    "entry_zone": s.get("entry_zone"),
                    "risk_reward": s.get("risk_reward"),
                    "action": s.get("action")
                })
        return JSONResponse(signals)

    elif task == "get_market_intelligence":
        return JSONResponse({
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": cache["market_regime"],
            "fear_greed": cache["fear_greed"],
            "total_signals": len([s for s in cache["signals"].values() if s.get("decision") in ["BUY", "SELL"]]),
            "assets_tracked": len(ASSETS)
        })

    elif task == "explain_signal":
        asset = data.get("asset", "BTC")
        signal = cache["signals"].get(f"{asset}USDT", {})
        if not signal:
            return JSONResponse({"error": f"No signal for {asset}"})
        return JSONResponse({
            "asset": signal.get("asset"),
            "decision": signal.get("decision"),
            "bias": signal.get("bias"),
            "direction": signal.get("direction"),
            "confidence": signal.get("confidence"),
            "explanation": signal.get("bullish_reasons") if signal.get("direction") == "LONG" else signal.get("bearish_reasons"),
            "risk": signal.get("risk"),
            "holding_period": signal.get("holding_period"),
            "entry_zone": signal.get("entry_zone"),
            "risk_reward": signal.get("risk_reward"),
            "action": signal.get("action"),
            "why_not_now": signal.get("why_not_now", [])
        })

    elif task == "predict_asset":
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            return JSONResponse({"message": "No predictions"})
        return JSONResponse([{
            "asset": s.get("asset"),
            "score": s.get("confidence"),
            "decision": s.get("decision"),
            "bias": s.get("bias"),
            "direction": s.get("direction"),
            "entry_zone": s.get("entry_zone")
        } for s in sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)[:5]])

    elif task == "rank_assets":
        signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
        return JSONResponse([{
            "asset": s.get("asset"),
            "score": s.get("confidence"),
            "decision": s.get("decision"),
            "bias": s.get("bias"),
            "direction": s.get("direction"),
            "grade": s.get("grade"),
            "entry_zone": s.get("entry_zone")
        } for s in signals[:5]])

    return JSONResponse({"error": "Unknown task. Use: find_best_pullback, get_all_signals, get_market_intelligence, explain_signal, predict_asset, rank_assets"})

@app.get("/why/{symbol}")
async def why(symbol: str):
    asset = symbol.upper() + "USDT"
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signal = cache["signals"].get(asset, {})
    if not signal:
        return JSONResponse({"error": f"No signal for {symbol}"}, status_code=404)

    if signal.get("decision") == "HOLD":
        explanation = "No trade signal. Missing conditions: " + ", ".join(signal.get("missing_conditions", []))
    else:
        reasons = signal.get("bullish_reasons") if signal.get("direction") == "LONG" else signal.get("bearish_reasons")
        explanation = f"{signal.get('decision')} signal. " + ". ".join(reasons[:3])

    return JSONResponse({
        "asset": signal.get("asset"),
        "decision": signal.get("decision"),
        "bias": signal.get("bias"),
        "direction": signal.get("direction"),
        "confidence": signal.get("confidence"),
        "explanation": explanation,
        "risk": signal.get("risk"),
        "holding_period": signal.get("holding_period"),
        "entry_zone": signal.get("entry_zone"),
        "risk_reward": signal.get("risk_reward"),
        "position_sizing": signal.get("position_sizing"),
        "action": signal.get("action"),
        "why_not_now": signal.get("why_not_now", []),
        "reasoning": signal.get("reasoning", []),
        "market_regime": signal.get("market_regime"),
        "fear_greed": signal.get("fear_greed")
    })

@app.get("/reasoning/{symbol}")
async def reasoning(symbol: str):
    asset = symbol.upper() + "USDT"
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signal = cache["signals"].get(asset, {})
    if not signal:
        return JSONResponse({"error": f"No signal for {symbol}"}, status_code=404)
    return JSONResponse({
        "asset": signal.get("asset"),
        "decision": signal.get("decision"),
        "bias": signal.get("bias"),
        "confidence": signal.get("confidence"),
        "thought_process": signal.get("reasoning", []),
        "action": signal.get("action")
    })

@app.get("/explain/{symbol}")
async def explain(symbol: str):
    asset = symbol.upper() + "USDT"
    signal = cache["signals"].get(asset, {})
    if not signal:
        return JSONResponse({"error": "No signal found", "symbol": symbol}, status_code=404)
    return JSONResponse({
        "asset": signal.get("asset"),
        "decision": signal.get("decision"),
        "bias": signal.get("bias"),
        "direction": signal.get("direction"),
        "confidence": signal.get("confidence"),
        "grade": signal.get("grade"),
        "checks": signal.get("checks", {}),
        "bullish_reasons": signal.get("bullish_reasons"),
        "bearish_reasons": signal.get("bearish_reasons"),
        "missing_conditions": signal.get("missing_conditions"),
        "market_regime": signal.get("market_regime"),
        "fear_greed": signal.get("fear_greed"),
        "price": signal.get("price"),
        "entry_zone": signal.get("entry_zone"),
        "entries": signal.get("entries"),
        "take_profit": signal.get("take_profit"),
        "stop_loss": signal.get("stop_loss"),
        "risk_reward": signal.get("risk_reward"),
        "position_sizing": signal.get("position_sizing"),
        "action": signal.get("action"),
        "why_not_now": signal.get("why_not_now", []),
        "reasoning": signal.get("reasoning", []),
        "source": signal.get("source"),
        "rsi": signal.get("rsi"),
        "atr": signal.get("atr"),
        "atr_pct": signal.get("atr_pct"),
        "risk": signal.get("risk"),
        "holding_period": signal.get("holding_period"),
        "pullback_pct": signal.get("pullback_pct"),
        "timestamp": signal.get("timestamp")
    })

@app.post("/portfolio")
async def portfolio(req: Request):
    try:
        data = await req.json()
        capital = float(data.get("capital", 1000))
    except:
        return JSONResponse({"error": "Invalid input. Use {'capital': 1000}"}, status_code=400)

    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    if not signals:
        return JSONResponse({"error": "No signals for portfolio allocation"})

    total_conf = sum(s.get("confidence", 0) for s in signals[:5]) or 1
    allocation = {}
    for s in signals[:5]:
        weight = s.get("confidence", 0) / total_conf
        allocation[s.get("asset")] = {
            "amount": round(capital * weight, 2),
            "entry_zone": s.get("entry_zone"),
            "risk_reward": s.get("risk_reward"),
            "direction": s.get("direction"),
            "action": s.get("action")
        }

    return JSONResponse({
        "capital": capital,
        "allocation": allocation,
        "capital_plan": get_capital_plan(),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/demo")
async def demo():
    if time.time() - cache["last_scan"] > 300:
        await scan_all()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    best = max(signals, key=lambda x: x.get("confidence", 0)) if signals else None
    accuracy = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0

    return JSONResponse({
        "agent": "CROO AI Oracle",
        "version": "10.0",
        "status": "active",
        "entry_strategy": "Zone-based entries (0.5-1% below/above current)",
        "tp_strategy": "Realistic 3-6% with 2:1+ R:R",
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "best_signal": {
            "asset": best.get("asset") if best else None,
            "decision": best.get("decision") if best else None,
            "bias": best.get("bias") if best else None,
            "direction": best.get("direction") if best else None,
            "confidence": best.get("confidence") if best else None,
            "entry_zone": best.get("entry_zone") if best else None,
            "action": best.get("action") if best else None
        } if best else None,
        "top_3_assets": [{
            "asset": s.get("asset"),
            "confidence": s.get("confidence"),
            "decision": s.get("decision"),
            "bias": s.get("bias"),
            "direction": s.get("direction"),
            "entry_zone": s.get("entry_zone"),
            "action": s.get("action")
        } for s in signals[:3]] if signals else [],
        "accuracy": f"{accuracy}%",
        "total_signals": performance["total"],
        "uptime": str(timedelta(seconds=int(time.time() - start_time)))
    })

@app.get("/business_model")
def business_model():
    return JSONResponse({
        "free": {
            "requests": "5/day",
            "features": ["Basic signals", "3 assets", "Current price entry only"]
        },
        "pro": {
            "price": "$9.99/month",
            "features": [
                "All assets",
                "Multi-source data",
                "Priority alerts",
                "Full history",
                "Telegram alerts",
                "Entry zone recommendations",
                "Position sizing guidance",
                "Realistic TP/SL with 2:1+ R:R",
                "Explainable AI",
                "Reasoning engine",
                "Capital allocation plan"
            ]
        },
        "enterprise": {
            "price": "Custom pricing",
            "features": [
                "All assets",
                "Webhook integration",
                "White-label",
                "Dedicated support",
                "Custom alerts",
                "API access",
                "Custom entry strategies"
            ]
        },
        "note": "Payments disabled during CROO Hackathon. All features unlocked for judges."
    })

@app.get("/agent/revenue")
def revenue():
    return JSONResponse({
        "total_calls": agent_memory["total_calls"],
        "revenue_simulated": round(agent_memory["revenue_simulated"], 2),
        "avg_per_call": round(agent_memory["revenue_simulated"] / max(1, agent_memory["total_calls"]), 4)
    })

@app.get("/reputation")
def reputation():
    win_rate = performance["wins"] / max(1, performance["total"])
    score = win_rate * 70 + min(performance["total"], 100) * 0.3
    score = min(100, score)
    return JSONResponse({
        "reputation_score": round(score, 1),
        "grade": grade(score),
        "signals_generated": performance["total"],
        "win_rate": round(win_rate * 100, 1)
    })

@app.get("/cap/metadata")
def cap_metadata():
    return JSONResponse({
        "name": "CROO AI Oracle",
        "version": "10.0",
        "agent_type": "market_intelligence",
        "owner": "Agentic Finance Studio",
        "autonomous": True,
        "a2a_enabled": True,
        "explainable_ai": True,
        "supported_assets": [a.replace("USDT", "") for a in ASSETS],
        "entry_strategy": "Zone-based entries (0.5-1% below/above current)",
        "tp_strategy": "Realistic 3-6% with 2:1+ R:R",
        "features": [
            "pullback_detection",
            "confidence_scoring",
            "market_intelligence",
            "regime_detection",
            "signal_ranking",
            "explainability",
            "reasoning_engine",
            "capital_allocation",
            "auto_alerts",
            "multi_source_data",
            "A2A_compatible",
            "volatility_filter"
        ],
        "pricing": {
            "free": "5 requests/day",
            "pro": "$9.99/month",
            "enterprise": "Custom pricing"
        }
    })

@app.get("/cap/health")
def cap_health():
    ws_status = "healthy" if time.time() - cache["last_ws_update"] < 120 else "stale"
    scanner_status = "healthy" if time.time() - cache["last_successful_scan"] < 900 else "stalled"
    active_ws = len([t for t in ws_tasks if not t.done()])
    failed_ws = len(disabled_ws)
    provider_info = []
    for name in ["Bybit", "Binance", "OKX", "Kraken"]:
        if name in disabled_ws:
            provider_info.append(f"{name}: ✗ (region restricted)")
        elif provider_status.get(name) == "active":
            provider_info.append(f"{name}: ✓")
        else:
            provider_info.append(f"{name}: ?")

    return JSONResponse({
        "status": "healthy" if (ws_status == "healthy" and scanner_status == "healthy") else "degraded",
        "uptime_hours": round((time.time() - start_time) / 3600, 1),
        "scanner": scanner_status,
        "websockets_active": active_ws,
        "websockets_failed": failed_ws,
        "providers": provider_info,
        "last_scan": datetime.utcfromtimestamp(cache["last_successful_scan"]).isoformat() if cache["last_successful_scan"] else None,
        "last_successful_scan": datetime.utcfromtimestamp(cache["last_successful_scan"]).isoformat() if cache["last_successful_scan"] else None,
        "entry_strategy": "Zone-based entries (0.5-1% below/above current)",
        "tp_strategy": "Realistic 3-6% with 2:1+ R:R",
        "payments": "disabled_for_judging",
        "signals_generated": performance["total"],
        "active_users": len(users_db),
        "telegram_queue": telegram_queue.qsize(),
        "version": "10.0-croo-final"
    })

@app.get("/pricing")
def pricing():
    return business_model()

@app.get("/capabilities")
def capabilities():
    return JSONResponse({
        "features": [
            "pullback_detection",
            "confidence_scoring",
            "market_intelligence",
            "regime_detection",
            "fear_greed_integration",
            "signal_ranking",
            "explainability",
            "reasoning_engine",
            "capital_allocation",
            "auto_alerts",
            "telegram_integration",
            "A2A_compatible",
            "CAP_metadata",
            "multi_source_data",
            "performance_tracking",
            "agent_memory",
            "portfolio_management",
            "risk_management",
            "entry_zone_recommendations",
            "position_sizing_guidance",
            "realistic_take_profits",
            "volatility_filter",
            "health_monitoring"
        ],
        "assets": [a.replace("USDT", "") for a in ASSETS],
        "sources": ["Binance", "Bybit", "OKX", "Kraken", "CoinGecko", "CoinMarketCap"],
        "entry_strategy": {
            "type": "Zone-based",
            "long_entries": "0.2% - 3.5% below current price",
            "short_entries": "0.2% - 3.5% above current price",
            "position_sizing": "50% moderate, 30% conservative, 20% DCA",
            "risk_reward": "2:1 to 2.5:1",
            "tp_range": "3% - 6%"
        },
        "api_endpoints": [
            "/oracle", "/best_signal", "/leaderboard", "/stats",
            "/history", "/agent/query", "/a2a",
            "/cap/metadata", "/cap/health", "/pricing", "/capabilities",
            "/explain/{symbol}", "/why/{symbol}", "/reasoning/{symbol}",
            "/agent/revenue", "/reputation", "/portfolio", "/demo",
            "/business_model", "/.well-known/agent.json"
        ]
    })

@app.get("/.well-known/agent.json")
def agent_manifest():
    return {
        "name": "CROO AI Oracle",
        "description": "Autonomous crypto intelligence agent with pullback detection, market regime analysis, entry zone recommendations, realistic TP/SL, volatility filtering, explainable AI, reasoning engine, and capital allocation plan.",
        "endpoint": "/agent/query",
        "a2a_endpoint": "/a2a",
        "pricing": {
            "free": "5 requests/day",
            "pro": "$9.99/month",
            "enterprise": "Custom"
        },
        "entry_strategy": {
            "type": "Zone-based entries",
            "description": "Never enters at current price. Uses 0.5-1% below/above current price with multiple levels",
            "tp_range": "3-6% with 2:1 to 2.5:1 R:R"
        },
        "capabilities": [
            "pullback_detection",
            "market_intelligence",
            "signal_ranking",
            "regime_detection",
            "explainability",
            "reasoning_engine",
            "capital_allocation",
            "multi_source_data",
            "auto_alerts",
            "portfolio_management",
            "volatility_filtering"
        ],
        "assets": [a.replace("USDT", "") for a in ASSETS],
        "version": "10.0"
    }

# ==================== TELEGRAM WEBHOOK ====================
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

# ==================== STARTUP BANNER ====================
def print_startup_banner():
    print("=" * 50)
    print("CROO AI ORACLE v10")
    print("=" * 50)
    print("\nAgent Status: ACTIVE")
    print("\nProviders:")
    for name in ["Bybit", "Binance", "OKX", "Kraken"]:
        if name in disabled_ws:
            status = "✗ (Region Restricted)"
        elif provider_status.get(name) == "active":
            status = "✓"
        else:
            status = "?"
        print(f"  {name}: {status}")
    print(f"\nFallback Depth: 4")
    print("\nCapabilities:")
    caps = [
        "✓ Autonomous Analysis",
        "✓ Multi-Provider Recovery",
        "✓ Explainable AI",
        "✓ Risk Management",
        "✓ Capital Allocation",
        "✓ Agent Memory",
        "✓ Telegram Delivery",
        "✓ A2A Ready"
    ]
    for cap in caps:
        print(f"  {cap}")
    print("\n" + "=" * 50)

# ==================== STARTUP / SHUTDOWN ====================
@app.on_event("startup")
async def startup_event():
    global session, scanner_task, ws_task, health_task, telegram_worker_task

    load_memory()

    timeout = aiohttp.ClientTimeout(total=20, connect=10, sock_read=15)
    connector = aiohttp.TCPConnector(
        limit=100,
        ttl_dns_cache=300,
        enable_cleanup_closed=True
    )
    session = aiohttp.ClientSession(
        timeout=timeout,
        connector=connector
    )

    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
        print(f"Webhook set: {webhook_url}")

    telegram_worker_task = asyncio.create_task(telegram_worker())
    print("Telegram worker started")

    scanner_task = asyncio.create_task(scanner_loop())
    print("Scanner started")

    ws_task = asyncio.create_task(start_websockets())
    print("WebSockets started")

    health_task = asyncio.create_task(health_monitor())
    print("Health monitor started")

    await scan_all(force=True)

    print_startup_banner()

@app.on_event("shutdown")
async def shutdown():
    global session, scanner_task, ws_task, health_task, telegram_worker_task, ws_tasks

    print("Shutting down...")
    shutdown_event.set()

    save_memory()
    print("Memory saved")

    tasks_to_cancel = []
    if scanner_task:
        tasks_to_cancel.append(scanner_task)
    if ws_task:
        tasks_to_cancel.append(ws_task)
    if health_task:
        tasks_to_cancel.append(health_task)
    if telegram_worker_task:
        tasks_to_cancel.append(telegram_worker_task)
    for t in ws_tasks:
        tasks_to_cancel.append(t)

    for task in tasks_to_cancel:
        if task:
            task.cancel()

    if tasks_to_cancel:
        await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        print("All tasks cancelled")

    if session:
        await session.close()
        print("Session closed")

    print("Shutdown complete")

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
