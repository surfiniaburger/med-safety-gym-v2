"""
Telegram Bridge for SafeClaw Agent

Converts Telegram messages → A2A protocol → SafeClaw Agent → Guardian responses
"""

import os
import re
import tempfile
import logging
import asyncio
from typing import Optional
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import html

from a2a.types import Message, Part, TextPart, TaskState
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.utils import new_task

from .claw_agent import SafeClawAgent
from .session_memory import SessionStore
from .voice import text_to_speech, cleanup_voice_note

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Silence noisy third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class TelegramUpdater:
    """Consolidated A2A TaskUpdater for Telegram with support for direct queries."""
    def __init__(self, callback_query=None):
        self.callback_query = callback_query
        self.responses = []
        self.is_failed = False
        self.state = None
        self.metadata = None

    async def update_status(self, state, message, metadata=None):
        self.state = state
        self.metadata = metadata
        if state == TaskState.failed:
            self.is_failed = True
        
        response_text = message.parts[0].root.text
        self.responses.append(response_text)
        
        # If this is a direct update (callback resume), reply immediately
        if self.callback_query:
            await self.callback_query.message.reply_text(html.escape(response_text), parse_mode='HTML')

    async def start_work(self): pass
    async def complete(self): pass
    async def failed(self, message):
        await self.update_status(TaskState.failed, message)

class TelegramBridge:
    """Bridges Telegram messages to SafeClaw A2A Agent."""
    
    def __init__(self, token: str):
        self.token = token
        self.agent = SafeClawAgent()
        self.sessions = SessionStore()  # Multi-user session memory
        self.app = Application.builder().token(token).build()
        
        # Register handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Portable temp directory for voice notes
        self.temp_dir = os.path.join(tempfile.gettempdir(), "safeclaw_voice")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message when /start is issued."""
        await update.message.reply_text(
            "🤖 **SafeClaw Guardian** is ready!\n\n"
            "I enforce **Entity Parity** safety checks on all medical actions.\n\n"
            "Try asking me to:\n"
            "• 'Check safety of: Prescribe Panobinostat for DIPG'\n"
            "• 'Verify: Administer ONC201'\n\n"
            "⚠️ I will **block** any action that introduces unknown clinical entities.",
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help information."""
        await update.message.reply_text(
            "**SafeClaw Commands**\n\n"
            "/start - Introduction\n"
            "/help - Show this message\n\n"
            "**How it works:**\n"
            "Send me any medical action, and I'll verify it against known entities "
            "using the MCP Entity Parity tool.\n\n"
            "Examples:\n"
            "• 'Prescribe Drug X for condition Y'\n"
            "• 'Check: Enroll patient in NCT12345'\n",
            parse_mode='Markdown'
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        user_text = update.message.text
        chat_id = update.effective_chat.id
        user_id = str(chat_id)  # Use chat_id as user identifier
        
        logger.info(f"Received message from {chat_id}: {user_text}")
        
        # Get or create session for this user
        session = self.sessions.get_or_create(user_id)
        session.add_message("user", user_text)
        
        # Send "thinking" indicator
        await update.message.reply_text("🧠 Checking with Guardian...", parse_mode='Markdown')
        
        try:
            # Convert to A2A Message
            a2a_message = Message(
                role="user",
                messageId=f"telegram-{update.message.message_id}",
                parts=[Part(root=TextPart(kind="text", text=user_text))]
            )
            
            # Create mock updater to capture agent responses
            responses = []
            
            updater = TelegramUpdater()
            
            # Run agent with session context
            await self.agent.run(a2a_message, updater, session=session)
            
            # Handle results (Toxic Context Prevention)
            if updater.is_failed:
                session.pop_message()
            elif updater.responses:
                session.add_message("assistant", "\n\n".join(updater.responses))
            
            # Final response formatting
            responses = updater.responses
            
            # Persist to database (SQLite)
            self.sessions.save(session)
            
            # Send response back to Telegram
            if responses:
                final_response = "\n\n".join(responses)
            else:
                final_response = "✅ Request processed."
            
            # Use HTML for better escaping of repo names with underscores
            if getattr(updater, 'state', None) == TaskState.input_required:
                keyboard = [
                    [
                        InlineKeyboardButton("✅ Confirm", callback_data="approve"),
                        InlineKeyboardButton("❌ Reject", callback_data="reject"),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(html.escape(final_response), reply_markup=reply_markup, parse_mode='HTML')
                
                # Store pending action for resumption
                session.pending_action = updater.metadata
                self.sessions.save(session)
            else:
                await update.message.reply_text(html.escape(final_response), parse_mode='HTML')
            
            # VOICE LOGIC:
            # 1. Automatic voice for Safety Failures (Guardian Alerts)
            # 2. Manual voice for "!v" or "say" keywords
            should_voice = updater.is_failed or any(k in user_text.lower() for k in ["!v ", "say ", "voice "])
            
            if should_voice and responses:
                await self.send_voice_note(update, final_response)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ <b>Error:</b> {html.escape(str(e))}",
                parse_mode='HTML'
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks for confirmation/rejection."""
        query = update.callback_query
        user_id = str(update.effective_chat.id)
        session = self.sessions.get_or_create(user_id)
        
        await query.answer()
        
        if query.data == "approve" and session.pending_action:
            action = session.pending_action
            tool_name = action["tool_name"]
            tool_args = action["tool_args"]
            
            await query.edit_message_text(f"⏳ <b>Executing:</b> {html.escape(tool_name)}...", parse_mode='HTML')
            
            await self.agent.execute_confirmed_tool(tool_name, tool_args, TelegramUpdater(query), session=session)
            session.pending_action = None
        else:
            await query.edit_message_text("❌ <b>Action Rejected.</b>", parse_mode='HTML')
            session.pending_action = None
            
        self.sessions.save(session)

    async def send_voice_note(self, update: Update, text: str):
        """Generate and send a voice note to the user."""
        chat_id = update.effective_chat.id
        msg_id = update.message.message_id
        file_path = os.path.join(self.temp_dir, f"voice_{chat_id}_{msg_id}.mp3")
        
        # Clean text for TTS (remove all HTML tags using regex)
        clean_text = re.sub(r"<[^>]*>", "", text).replace("🚨", "Alert!").replace("❌", "Denied.")
        
        await update.message.reply_chat_action("record_voice")
        
        success = await text_to_speech(clean_text, file_path)
        if success:
            try:
                with open(file_path, "rb") as audio:
                    await update.message.reply_voice(
                        voice=audio,
                        caption="🎙️ Audio Safety Alert" if "BLOCK" in text or "Denied" in text else "🎙️ Voice Note"
                    )
            finally:
                await cleanup_voice_note(file_path)
        else:
            logger.error("Failed to generate voice note.")
    
    def run(self):
        """Start the Telegram bot."""
        logger.info("🚀 Starting SafeClaw Telegram Bridge...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Entry point for Telegram bridge."""
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not found in environment!")
        logger.error("Please set it in .env or export it:")
        logger.error("  export TELEGRAM_BOT_TOKEN='your-bot-token-here'")
        return
    
    bridge = TelegramBridge(token)
    bridge.run()


if __name__ == "__main__":
    main()
