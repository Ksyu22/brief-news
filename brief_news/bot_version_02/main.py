import os
import functions_framework
from telegram import Bot, Update
from telegram.ext import CommandHandler, Dispatcher, MessageHandler, Filters
import constants.messages as MESSAGES
import constants.commands as COMMANDS
import constants.params as PARAMS
from helpers.get_info import get_info

TOKEN = os.environ.get("BOT_TOKEN")
bot = Bot(token = TOKEN)

dispatcher = Dispatcher(bot = bot, update_queue = None, workers = 0)

def start(update, context):
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id = chat_id, text = MESSAGES.MSG_ON_START, disable_web_page_preview = PARAMS.IS_DISABLE_WEB_PREVIEW)

def get_summary_info(update, context):
    '''This is to obtain the news summary for the chosen category and format before presenting.'''
    news_df = get_info(update.message.text)

    print (type(news_df))
    print (news_df)
    if news_df is None or len(news_df) == 0:
        update.message.reply_text(MESSAGES.MSG_ON_START, disable_web_page_preview = PARAMS.IS_DISABLE_WEB_PREVIEW)
        return

    result_1, result_2 = '', ''
    result_1 = result_1 + f"<b><u>{news_df[0]['title']}</u></b>" + '\n\n' + news_df[0]['summary_text']

    if len(news_df) >= 2:
        result_2 = result_2 + f"<b><u>{news_df[1]['title']}</u></b>" + '\n\n' + news_df[1]['summary_text']

    # If the result is invalid, return the custom response from GetInfo() and exit the function
    if result_1.__class__ is str:
        update.message.reply_text(result_1, parse_mode = PARAMS.CURRENT_PARSE_MODE, disable_web_page_preview = PARAMS.IS_DISABLE_WEB_PREVIEW)
        if len(news_df) >= 2:
            update.message.reply_text(result_2, parse_mode = PARAMS.CURRENT_PARSE_MODE, disable_web_page_preview = PARAMS.IS_DISABLE_WEB_PREVIEW)
    return

dispatcher.add_handler(CommandHandler(COMMANDS.START, start))
dispatcher.add_handler(MessageHandler(Filters.text, get_summary_info))

# [START main handler for cloud function]
@functions_framework.http
def handler(request):
    update = Update.de_json(request.get_json(force = True), bot)
    dispatcher.process_update(update)
    return MESSAGES.MSG_RESPONSE_OK
