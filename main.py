import os
import asyncio
from telegram.ext import Application, CommandHandler

TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update, context):
    await update.message.reply_text("CROO Oracle is live ✅")

def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN not found in environment")
    
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    print("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
