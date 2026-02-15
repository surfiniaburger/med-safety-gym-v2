"""
Telegram Bridge for SafeClaw Agent

Converts Telegram messages → A2A protocol → SafeClaw Agent → Guardian responses
"""

import os
import logging
import asyncio
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from a2a.types import Message, Part, TextPart, TaskState
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.utils import new_task

from .claw_agent import SafeClawAgent
from .session_memory import SessionStore

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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
            
            class TelegramUpdater:
                """Mock updater that captures responses for Telegram."""
                def __init__(self):
                    self.is_failed = False

                async def update_status(self, state, message):
                    if state == TaskState.failed:
                        self.is_failed = True
                    response_text = message.parts[0].root.text
                    responses.append(response_text)
                
                async def start_work(self):
                    pass
                
                async def complete(self):
                    pass
                
                async def failed(self, message):
                    self.is_failed = True
                    response_text = message.parts[0].root.text
                    responses.append(response_text)
            
            updater = TelegramUpdater()
            
            # Run agent with session context
            await self.agent.run(a2a_message, updater, session=session)
            
            # Handle results
            if updater.is_failed:
                # Toxic Context Prevention: Roll back the history for this session
                # because the action was unsafe. We don't want to learn from it.
                session.pop_message()
            else:
                # Commit: Add assistant response to history
                if responses:
                    session.add_message("assistant", "\n\n".join(responses))
            
            # Send response back to Telegram
            if responses:
                final_response = "\n\n".join(responses)
            else:
                final_response = "✅ Request processed."
            
            await update.message.reply_text(final_response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Error: {str(e)}",
                parse_mode='Markdown'
            )
    
    def run(self):
        """Start the Telegram bot."""
        logger.info("🚀 Starting SafeClaw Telegram Bridge...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Entry point for Telegram bridge."""
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
