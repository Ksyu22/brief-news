"""Readme :  pip install python-telegram-bot --upgrade"""
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from config import BOT_TOKEN
from dictionary import get_info



def start(update, context):
    '''This is to set up the introductory statement for the bot when the /start command is invoked.'''
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id=chat_id, text="Hello there. Provide any English word and I will give you a bunch "
                                                   "of information about it.")



def get_word_info(update, context):
    '''This is to obtain the information of the word provided and format before presenting.'''
    # get the word info
    word_info = get_info(update.message.text)

    # If the user provides an invalid English word, return the custom response from get_info() and exit the function
    if word_info.__class__ is str:
        update.message.reply_text(word_info)
        return

    # get the word the user provided
    word = word_info['word']

    message = f"Word: {word}"

    update.message.reply_text(message)


def main():
    '''Main app function which runs constantly when this .py file is executed.'''

    telegram_bot_token = BOT_TOKEN

    updater = Updater(token=telegram_bot_token, use_context=True)
    dispatcher = updater.dispatcher

    # run the start function when the user invokes the /start command
    dispatcher.add_handler(CommandHandler("start", start))

    # invoke the get_word_info function when the user sends a message
    # that is not a command.
    dispatcher.add_handler(MessageHandler(Filters.text, get_word_info))

    updater.start_polling()
    updater.idle()

    # updater.start_webhook(listen="0.0.0.0",
    #                     port=int(os.environ.get('PORT', 5000)),
    #                     url_path=telegram_bot_token,
    #                     webhook_url=  + telegram_bot_token
    #                     )


if __name__ == "__main__":
    main()
