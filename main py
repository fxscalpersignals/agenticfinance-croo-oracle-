import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from collections import deque

import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler

# ================= FASTAPI CAP WRAPPER =================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_CHAT_ID = os.getenv("CHAT_ID")
if ADMIN_CHAT_ID:
    try:
        ADMIN_CHAT_ID = int(ADMIN_CHAT_ID)
    except ValueError:
        ADMIN_CHAT_ID = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

HEADERS = {"User-Agent": "AgenticFinanceStudio/3.0-CROO"}
TIMEOUT = aiohttp.ClientTimeout(total=15)

COINS = ["btc", "eth", "xrp", "sol"]
COIN_NAMES = {"btc": "bitcoin", "eth": "ethereum", "xrp": "ripple", "sol": "solana"}

SECTORS = {
    "AI": ["eth", "sol"],
    "PAYFI": ["xrp"],
    "RWA": ["btc"],
    "DEFI": ["eth", "sol"]
}

performance_log = {coin: deque(maxlen=100) for coin in COINS}
user_last_interaction = {}
price_cache = {}
rsi_cache = {}
paper_positions = {}

# ================= SAFETY FOR MONETIZATION =================
api_semaphore = asyncio.Semaphore(5)
REQUEST_DELAY = 0.6

analytics = {"signals_generated": 0, "alerts_sent": 0, "scanner_alerts": 0}
start_time = datetime.now()

BYBIT_REFERRAL_LINK = "https://www.bybit.com/invite?ref=N8GY3B&medium=referral&utm_campaign=evergreen"
PORTFOLIO_FILE = "paper_portfolio.json"

if os.path.exists(PORTFOLIO_FILE):
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            paper_positions = json.load(f)
    except:
        paper_positions = {}

def save_portfolio():
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(paper_positions, f)
    except:
        pass

def format_price(price: float) -> str:
    if price < 1:
        return f"{price:.6f}"
    elif price < 100:
        return f"{price:.2f}"
    return f"{price:,.0f}"

def build_main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 BTC", callback_data="signal_btc"), InlineKeyboardButton("📊 ETH", callback_data="signal_eth")],
        [InlineKeyboardButton("📊 XRP", callback_data="signal_xrp"), InlineKeyboardButton("📊 SOL", callback_data="signal_sol")],
        [InlineKeyboardButton("📈 Sector Map", callback_data="sectors"), InlineKeyboardButton("🐳 Whale Radar", callback_data="whale")],
        [InlineKeyboardButton("📊 ETF Flows", callback_data="etf"), InlineKeyboardButton("🧠 Intelligence", callback_data="news")],
        [InlineKeyboardButton("📈 Performance", callback_data="performance"), InlineKeyboardButton("📊 Stats", callback_data="stats")],
        [InlineKeyboardButton("🔄 Scanner Status", callback_data="scanner_status"), InlineKeyboardButton("💰 Trade Now", callback_data="trade_now")],
        [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ================= GLOBAL SESSION + SHUTDOWN =================
async def get_session(app):
    if "session" not in app.bot_data:
        connector = aiohttp.TCPConnector(limit=15, ttl_dns_cache=300, force_close=False)
        app.bot_data["session"] = aiohttp.ClientSession(timeout=TIMEOUT, connector=connector)
    return app.bot_data["session"]

async def post_shutdown(app):
    session = app.bot_data.get("session")
    if session and not session.closed:
        await session.close()
    save_portfolio()
    logging.info("✅ Graceful shutdown completed.")

# ================= 3-TIER FALLBACK DATA SYSTEM =================
async def get_price_binance(session, coin):
    """PRIMARY: Binance - 1200 req/min, no key"""
    try:
        symbol = f"{coin.upper()}USDT"
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        async with session.get(url, timeout=3) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["lastPrice"]), float(data["priceChangePercent"]), "🥇 Binance"
    except: pass
    return None, None, None

async def get_price_mexc(session, coin):
    """SECONDARY: MEXC - no key, no rate limit issues"""
    try:
        symbol = f"{coin.upper()}_USDT"
        url = f"https://api.mexc.com/api/v3/ticker/price?symbol={symbol}"
        async with session.get(url, timeout=3) as r:
            if r.status == 200:
                price = float((await r.json())['price'])
                # MEXC doesn't give 24h change in this endpoint, so we skip change
                return price, 0.0, "🥈 MEXC"
    except: pass
    return None, None, None

async def get_price_coingecko(session, coin):
    """TERTIARY: CoinGecko - 30 req/min, no key"""
    try:
        cg_id = COIN_NAMES[coin]
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd&include_24hr_change=true"
        async with session.get(url, timeout=3) as r:
            if r.status == 200:
                data = await r.json()
                if cg_id in data:
                    return data[cg_id]["usd"], float(data[cg_id].get("usd_24h_change", 0)), "🥉 CoinGecko"
    except: pass
    return None, None, None

async def fetch_price(session, symbol):
    """Tries Binance → MEXC → CoinGecko. Returns first that works."""
    async with api_semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        
        for func in [get_price_binance, get_price_mexc, get_price_coingecko]:
            price, change, source = await func(session, symbol)
            if price:
                confidence = 95 if "Binance" in source else 80 if "MEXC" in source else 70
                logging.info(f"Got {symbol} from {source}")
                return {"price": price, "change": change, "source": source, "confidence": confidence}
    
    return {"price": None, "change": 0.0, "source": "All_Failed", "confidence": 0}

async def get_cached_price(session, symbol):
    now = datetime.now()
    if symbol in price_cache and now - price_cache[symbol]["time"] < timedelta(seconds=25):
        return price_cache[symbol]["data"]
    data = await fetch_price(session, symbol)
    price_cache[symbol] = {"data": data, "time": now}
    return data

async def get_atr(session, symbol, period=14):
    try:
        async with api_semaphore:
            await asyncio.sleep(REQUEST_DELAY)
            async with session.get(f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}USDT&interval=1h&limit={period+2}", timeout=TIMEOUT) as resp:
                data = await resp.json()
                if len(data) < period + 1: return None
                trs = []
                for i in range(1, len(data)):
                    high = float(data[i][2])
                    low = float(data[i][3])
                    prev = float(data[i-1][4])
                    tr = max(high - low, abs(high - prev), abs(low - prev))
                    trs.append(tr)
                return sum(trs) / len(trs)
    except:
        return None

async def get_rsi(session, symbol):
    key = f"rsi_{symbol}"
    now = datetime.now()
    if key in rsi_cache and now - rsi_cache[key]["time"] < timedelta(seconds=180):
        return rsi_cache[key]["value"]
    try:
        async with api_semaphore:
            await asyncio.sleep(REQUEST_DELAY)
            async with session.get(f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}USDT&interval=1h&limit=20", timeout=TIMEOUT) as resp:
                data = await resp.json()
                if len(data) < 15: return None
                closes = [float(c[4]) for c in data]
                gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
                losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14 if sum(losses[-14:]) > 0 else 0.0001
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                rsi_value = round(rsi, 1)
                rsi_cache[key] = {"value": rsi_value, "time": now}
                return rsi_value
    except:
        return None

async def get_ema(session, symbol, period=20):
    try:
        async with api_semaphore:
            await asyncio.sleep(REQUEST_DELAY)
            async with session.get(f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}USDT&interval=1h&limit={period+10}", timeout=TIMEOUT) as resp:
                data = await resp.json()
                if len(data) < period: return None
                closes = [float(c[4]) for c in data]
                multiplier = 2 / (period + 1)
                ema = closes[0]
                for price in closes[1:]:
                    ema = (price * multiplier) + (ema * (1 - multiplier))
                return round(ema, 4)
    except:
        return None

# ================= SIGNAL WITH EMA =================
async def generate_signal(session, symbol):
    data = await get_cached_price(session, symbol)
    if data["price"] is None:
        return None

    rsi = await get_rsi(session, symbol)
    atr = await get_atr(session, symbol) or (data["price"] * 0.015)
    ema20 = await get_ema(session, symbol, 20)
    ema50 = await get_ema(session, symbol, 50)

    price = data["price"]
    change = data["change"]

    bullish_trend = ema20 and ema50 and ema20 > ema50
    bearish_trend = ema20 and ema50 and ema20 < ema50

    direction = "LONG" if change > 1.5 and bullish_trend else "SHORT" if change < -1.5 and bearish_trend else "NEUTRAL"

    if direction == "LONG":
        sentiment, emoji = "Bullish", "🚀"
        entry = price * 0.992
        sl = entry - (atr * 1.2)
        tp = entry + (atr * 2.5)
    elif direction == "SHORT":
        sentiment, emoji = "Bearish", "📉"
        entry = price * 1.008
        sl = entry + (atr * 1.2)
        tp = entry - (atr * 2.5)
    else:
        sentiment, emoji = "Neutral", "🟡"
        entry = price * 0.995
        sl = entry - (atr * 0.8)
        tp = entry + (atr * 1.5)

    if rsi and rsi > 70:
        reasoning = "Overbought - Momentum Extension (RSI)"
    elif rsi and rsi < 30:
        reasoning = "Oversold Pullback Opportunity (RSI)"
    elif bullish_trend and direction == "LONG":
        reasoning = "Bullish EMA Trend (EMA20 > EMA50)"
    elif bearish_trend and direction == "SHORT":
        reasoning = "Bearish EMA Trend (EMA20 < EMA50)"
    else:
        reasoning = "Multi-source Technical Convergence"

    confidence = 58
    if abs(change) > 5: confidence += 22
    elif abs(change) > 3: confidence += 15
    if rsi and 35 < rsi < 70: confidence += 18
    if bullish_trend or bearish_trend: confidence += 15
    confidence = min(98, confidence)

    signal = {**data, "symbol": symbol.upper(), "entry": entry, "tp": tp, "sl": sl,
              "sentiment": sentiment, "emoji": emoji, "bias": f"{direction} Signal",
              "rsi": rsi, "confidence": confidence, "reasoning": reasoning, "timeframe": "1H"}

    analytics["signals_generated"] += 1
    performance_log[symbol].append(signal)
    return signal

# ================= ALL OTHER FUNCTIONS UNCHANGED =================
async def get_sector_map(session):
    sector_data = {}
    for sector, coins in SECTORS.items():
        changes = []
        for c in coins:
            data = await get_cached_price(session, c)
            if data.get("price") is not None:
                changes.append(data["change"])
        avg = sum(changes) / len(changes) if changes else 0
        sector_data[sector] = avg
    
    if not sector_data:
        return "📈 **Sector Intelligence Map**\n\n⚠️ Data temporarily unavailable."
    
    strongest = max(sector_data, key=sector_data.get)
    msg = "📈 **Sector Intelligence Map**\n\n"
    for sector, avg in sector_data.items():
        emoji = "🟢" if avg > 1.5 else "🔴" if avg < -1.5 else "🟡"
        msg += f"{'🔥' if sector == strongest else emoji} **{sector}**: {avg:+.2f}%{' (Strongest)' if sector == strongest else ''}\n"
    return msg

async def get_whale_radar(session):
    moves = []
    for symbol in COINS:
        data = await get_cached_price(session, symbol)
        if data.get("price") is not None:
            moves.append((symbol.upper(), data["change"]))
    moves.sort(key=lambda x: abs(x[1]), reverse=True)
    msg = "🐳 **Whale Radar (Top Movers)**\n\n"
    for symbol, change in moves:
        if abs(change) >= 4.0:
            msg += f"🔥 **{symbol}**: {change:+.2f}% (EXTREME)\n"
        elif abs(change) >= 2.0:
            msg += f"📈 {symbol}: {change:+.2f}%\n"
        else:
            msg += f"📉 {symbol}: {change:+.2f}%\n"
    return msg

async def fetch_etf_flows(session):
    btc = await get_cached_price(session, "btc")
    eth = await get_cached_price(session, "eth")
    bias = "Bullish on BTC" if btc.get("change", 0) > eth.get("change", 0) else "Bullish on ETH"
    return f"📊 **ETF Intelligence**\n\nBTC 24h: {btc.get('change',0):+.2f}%\nETH 24h: {eth.get('change',0):+.2f}%\n\nBias: **{bias}**"

async def get_intelligence_feed(session):
    total = 0
    count = 0
    strongest = "N/A"
    max_change = -100
    for symbol in COINS:
        data = await get_cached_price(session, symbol)
        if data.get("price") is not None:
            total += data["change"]
            count += 1
            if data["change"] > max_change:
                max_change = data["change"]
                strongest = symbol.upper()
    avg = total / count if count > 0 else 0
    try:
        data = await fetch_with_retry(session, "https://api.alternative.me/fng/")
        ssi = int(data["data"][0]["value"]) if data and data.get("data") else round(50 + avg * 7)
        mood = data["data"][0]["value_classification"] if data and data.get("data") else "Neutral"
    except:
        ssi = round(50 + avg * 7)
        mood = "Neutral"
    return f"🧠 **Market Intelligence Feed**\n\n**SSI Score**: {ssi}/100\nMood: **{mood}**\nAvg Change: {avg:+.2f}%\nStrongest: **{strongest.upper()}**"

async def get_performance():
    msg = "📈 **Performance Tracker**\n\n"
    for coin in COINS:
        msg += f"{coin.upper()}: {len(performance_log[coin])} signals\n"
    return msg

async def get_stats():
    uptime = str(datetime.now() - start_time).split('.')[0]
    return f"📊 **Live Stats**\n\nSignals Generated: {analytics['signals_generated']}\nScanner Alerts: {analytics['scanner_alerts']}\nUptime: {uptime}"

async def get_scanner_status():
    uptime = str(datetime.now() - start_time).split('.')[0]
    return f"🔄 **Scanner Status**\n\nActive: ✅ Running\nScanner Alerts: {analytics['scanner_alerts']}\nTotal Signals: {analytics['signals_generated']}\nUptime: {uptime}"

async def open_paper_trade(user_id, symbol, entry_price, side):
    if user_id not in paper_positions:
        paper_positions[user_id] = {}
    paper_positions[user_id][symbol] = {"entry": entry_price, "amount": 1.0, "side": side}
    save_portfolio()

async def close_all_trades(user_id):
    if user_id in paper_positions and paper_positions[user_id]:
        paper_positions[user_id].clear()
        save_portfolio()
        return "✅ All paper trades closed successfully."
    return "No active trades."

async def get_portfolio(user_id, session):
    if user_id not in paper_positions or not paper_positions[user_id]:
        return "💼 No active paper trades."
    msg = "💼 **Your Paper Portfolio**\n\n"
    total_pnl = 0
    for symbol, pos in paper_positions[user_id].items():
        current = await get_cached_price(session, symbol.lower())
        if current and current.get("price"):
            pnl = (current["price"] - pos["entry"]) * pos["amount"] if pos["side"] == "LONG" else (pos["entry"] - current["price"]) * pos["amount"]
            total_pnl += pnl
            msg += f"{symbol}: {pos['side']} @ ${format_price(pos['entry'])} | PnL: ${format_price(pnl)}\n"
    msg += f"\n**Total Unrealized PnL: ${format_price(total_pnl)}**"
    return msg

async def market_scanner(app):
    await asyncio.sleep(45)
    logging.info("🚀 Market Scanner started (Safe Mode)")
    while True:
        try:
            session = await get_session(app)
            for coin in COINS:
                sig = await generate_signal(session, coin)
                if sig and sig["confidence"] >= 72:
                    msg = f"🚨 **AUTO SCANNER ALERT**\n{sig['symbol']} {sig['bias']}\nConfidence: {sig['confidence']}%\n{sig['reasoning']}"
                    if ADMIN_CHAT_ID:
                        await app.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                        analytics["scanner_alerts"] += 1
                await asyncio.sleep(1.2)
            await asyncio.sleep(240)
        except Exception as e:
            logging.error(f"Scanner error: {e}")
            await asyncio.sleep(90)

# ================= BUTTON HANDLER =================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    now = datetime.now()

    if user_id in user_last_interaction and (now - user_last_interaction[user_id]) < timedelta(seconds=2):
        await query.answer("⏳ Please wait 2 seconds.", show_alert=True)
        return

    user_last_interaction[user_id] = now
    await query.answer()

    action = query.data
    session = await get_session(context.application)

    try:
        if action.startswith("signal_"):
            symbol = action.split("_")[1]
            sig = await generate_signal(session, symbol)
            if sig:
                await open_paper_trade(user_id, sig["symbol"], sig["entry"], sig["bias"].split()[0])
                keyboard = [[InlineKeyboardButton("💰 Trade on Bybit", url=BYBIT_REFERRAL_LINK)],
                            [InlineKeyboardButton("🔙 Back", callback_data="back_main")]]
                msg = f"🧠 **{sig['symbol']} SIGNAL** — {sig['confidence']}% Confidence\n\n" \
                      f"💰 Current: **${format_price(sig['price'])}**\n" \
                      f"🎯 Entry: **${format_price(sig['entry'])}** (Pullback)\n" \
                      f"🏆 TP: **${format_price(sig['tp'])}**\n" \
                      f"🛑 SL: **${format_price(sig['sl'])}**\n\n" \
                      f"{sig['emoji']} {sig['sentiment']} | RSI: {sig.get('rsi', 'N/A')}\n" \
                      f"🔍 {sig['reasoning']}\n🔗 {sig['source']}"
                await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

        elif action == "trade_now":
            keyboard = [[InlineKeyboardButton("💰 Trade on Bybit", url=BYBIT_REFERRAL_LINK)],
                        [InlineKeyboardButton("🔙 Back", callback_data="back_main")]]
            await query.edit_message_text("💰 **Quick Trade**\n\nOpen exchange below:", 
                                         reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

        elif action == "scanner_status":
            msg = await get_scanner_status()
            await query.edit_message_text(msg, reply_markup=build_main_menu(), parse_mode=ParseMode.MARKDOWN)

        elif action == "portfolio":
            msg = await get_portfolio(user_id, session)
            keyboard = [[InlineKeyboardButton("❌ Close All Trades", callback_data="close_all")],
                        [InlineKeyboardButton("🔙 Back", callback_data="back_main")]]
            await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

        elif action == "close_all":
            result = await close_all_trades(user_id)
            await query.edit_message_text(result, reply_markup=build_main_menu(), parse_mode=ParseMode.MARKDOWN)

        elif action in ["sectors", "whale", "etf", "news", "performance", "stats"]:
            if action == "sectors": msg = await get_sector_map(session)
            elif action == "whale": msg = await get_whale_radar(session)
            elif action == "etf": msg = await fetch_etf_flows(session)
            elif action == "news": msg = await get_intelligence_feed(session)
            elif action == "performance": msg = await get_performance()
            elif action == "stats": msg = await get_stats()
            await query.edit_message_text(msg, reply_markup=build_main_menu(), parse_mode=ParseMode.MARKDOWN)

        elif action == "back_main":
            await query.edit_message_text("Welcome back to main menu!", reply_markup=build_main_menu(), parse_mode=ParseMode.MARKDOWN)

    except BadRequest as e:
        if "Message is not modified" not in str(e):
            logging.error(f"Telegram error: {e}")
    except Exception as e:
        logging.error(f"Button handler error: {e}")
        try:
            await query.edit_message_text("⚠️ Temporary error. Try again.", reply_markup=build_main_menu())
        except:
            pass

# ================= CAP API ENDPOINT FOR CROO =================
app_api = FastAPI(title="Agentic Finance Alpha Oracle - CROO")
app_api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SignalRequest(BaseModel):
    symbol: str

@app_api.get("/health")
async def health():
    return {"status": "ok", "agent": "AgenticFinance Alpha Oracle", "tracks": ["DeFi", "Research"]}

@app_api.post("/get_signal")
async def get_signal_api(req: SignalRequest):
    symbol = req.symbol.lower()
    if symbol not in COINS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol. Use: {COINS}")
    
    # Create temp session for API call
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        sig = await generate_signal(session, symbol)
        if not sig:
            raise HTTPException(status_code=503, detail="Data sources unavailable")
        
        return {
            "symbol": sig["symbol"],
            "signal": sig["bias"].split()[0],
            "entry": round(sig["entry"], 6),
            "tp": round(sig["tp"], 6),
            "sl": round(sig["sl"], 6),
            "confidence": sig["confidence"],
            "rsi": sig["rsi"],
            "reasoning": sig["reasoning"],
            "data_source": sig["source"],
            "price": sig["price"]
        }

# ================= START =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 **Agentic Finance Studio - CROO Edition**\n\n"
        "A2A-Native Trading Oracle • CAP Integrated\n\n"
        "Real-time signals with pullback entries & 3-tier fallback data.\n\n"
        "⚠️ Not financial advice. Trade at your own risk.\n\n"
        "Tap any button below:",
        reply_markup=build_main_menu(),
        parse_mode=ParseMode.MARKDOWN
    )

def run_fastapi():
    uvicorn.run(app_api, host="0.0.0.0", port=8000, log_level="error")

def main():
    if not TOKEN:
        logging.error("❌ TELEGRAM_TOKEN not set!")
        return

    # Start FastAPI in background thread for CAP
    threading.Thread(target=run_fastapi, daemon=True).start()
    logging.info("✅ CAP API started on port 8000 - /get_signal ready for CROO")

    async def _internal_post_init(app):
        await get_session(app)
        app.bot_data["scanner_task"] = asyncio.create_task(market_scanner(app))
        logging.info("✅ CROO Production Version Loaded - CAP + 3-tier fallbacks")

    application = (
        ApplicationBuilder()
       .token(TOKEN)
       .post_init(_internal_post_init)
       .post_shutdown(post_shutdown)
       .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))

    logging.info("🚀 Agentic Finance Studio is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
