from random import randint
from telegram.ext import CommandHandler, Updater, Dispatcher, MessageHandler, Filters
import telegram

TOKEN = "token"
chat_id = 331799484

def greet(bot, update):
    greetings = ['bonjour', 'hola', 'hallo', 'sveiki', 'namaste', 'szia', 'halo', 'ciao']
    rint = randint(0, len(greetings) - 1)
    update.message.reply_text(greetings[rint]+ " !")

def paladin(bot, update):
    string_1 = "Hello! I am Paladin, a Bot made to talk with you.\n"
    string_2 = "You can try using the commands \n "
    string_3 = "/greet :  Paladin greets you \n"
    string_4 = "/paladin : To know about paladin\n"
    update.message.reply_text(string_1 + string_2 + string_3 + string_4)

def noCommand(bot, update):
    string = update.message.text
    update.message.reply_text("I have no idea what "+string+" means ")


def main():
    updater = Updater(token = TOKEN)
    dispatcher = updater.dispatcher

    handler = CommandHandler('greet', greet)
    dispatcher.add_handler(handler)

    handler = CommandHandler('paladin', paladin)
    dispatcher.add_handler(handler)

    message_handler = MessageHandler(Filters.text, noCommand)
    dispatcher.add_handler(message_handler)
    updater.start_polling()

if __name__ == '__main__':
    main()
