import telebot
import os
from random import randint
import apiai, json
from textblob import TextBlob
from profanity_check import predict, predict_prob
from flask import Flask, request



TOKEN = "TOKEN"
bot = telebot.TeleBot(TOKEN)
server = Flask(__name__)

@bot.message_handler(commands = ['greet'])
def greet(message):
    greetings = ['bonjour', 'hola', 'hallo', 'sveiki', 'namaste', 'szia', 'halo', 'ciao']
    rint = randint(0, len(greetings) - 1)
    bot.reply_to(message, greetings[rint]+ " !")

@bot.message_handler(commands = ['sentiment'])
def sentiment(message):
    text = message.text[10:]
    review = Query(text)+""
    bot.reply_to(message, review)

@bot.message_handler(commands = ['paladin'])
def paladin(message):
    string_1 = "Hello! I am Paladin, a Bot made to talk with you.\n"
    string_2 = "You can try using the commands \n "
    string_3 = "/greet :  Paladin greets you \n"
    string_4 = "/paladin : To know about paladin\n"
    string_5 = "/sentiment : To find the sentiment of a sentence\n"
    bot.reply_to(message, string_1 + string_2 + string_3 + string_4 + string_5)

@bot.message_handler(func = lambda msg : True if msg.text[0]!='/' else False, content_types = ['text'])
def noc(message):
    #bot.reply_to(message, message.text)
    request = apiai.ApiAI('DIALOGTOKEN').text_request()
    request.lang = 'en'
    request.session_id = 'BatlabAIBot'
    #send request to dialgoflow
    request.query = message.text
    responseJson = json.loads(request.getresponse().read().decode('utf-8'))
    response = responseJson['result']['fulfillment']['speech']
    if response:
        bot.reply_to(message, response+'')
    else:
        bot.reply_to(message, 'I dont know what '+str(request.query)+' means!')


def Extract(text):
    profanity = predict_prob([text])[0]
    sentiment = TextBlob(text)
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    return [polarity, subjectivity, profanity]

def Query(review):
     result = Extract(review)
     string_res = "Polarity: "+str(result[0])+"\n"+"Subjectivity : "
     string_res += str(result[1])+"\n"+"Profanity"+str(result[2])
     return string_res





@server.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://vast-wave.herokuapp.com/' + TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
