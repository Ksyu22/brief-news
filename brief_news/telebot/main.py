import os
from dotenv import load_dotenv
load_dotenv()
"""Readme :  pip install python-telegram-bot --upgrade"""
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from brief_news.telebot.dictionary import get_info



def start(update, context):
    '''This is to set up the introductory statement for the bot when the /start command is invoked.'''
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id=chat_id, text="Hello there. Kindly provide number of the news category that you're interested in today.\n"
                                                   "0-business, \n1-entertainment, \n2-general, \n3-health, \n4-science, \n5-sports, \n6-technology")



def get_summary_info(update, context):
    '''This is to obtain the news summary for the chosen category and format before presenting.'''
    news_df = get_info(update.message.text)

    result_1, result_2 = '', ''
    result_1 = result_1 + f"<b><u>{news_df['title'][0]}</u></b>" + '\n\n' + news_df['summary_text'][0]

    if len(news_df)>=2:
        result_2 = result_2 + f"<b><u>{news_df['title'][1]}</u></b>" + '\n\n' + news_df['summary_text'][1]

    # If the result is invalid, return the custom response from get_info() and exit the function
    if result_1.__class__ is str:
        update.message.reply_text(result_1, parse_mode='HTML')
        if len(news_df)>=2:
            update.message.reply_text(result_2, parse_mode='HTML')
    return



def main():
    '''Main app function which runs constantly when this .py file is executed.'''

    telegram_bot_token = os.environ.get("BOT_TOKEN")

    updater = Updater(token=telegram_bot_token, use_context=True)
    dispatcher = updater.dispatcher

    # run the start function when the user invokes the /start command
    dispatcher.add_handler(CommandHandler("start", start))

    # invoke the get_word_info function when the user sends a message
    # that is not a command.
    dispatcher.add_handler(MessageHandler(Filters.text, get_summary_info))

    updater.start_polling()
    updater.idle()

    # updater.start_webhook(listen="0.0.0.0",
    #                     port=int(os.environ.get('PORT', 5000)),
    #                     url_path=telegram_bot_token,
    #                     webhook_url=  + telegram_bot_token
    #                     )


if __name__ == "__main__":
    main()
