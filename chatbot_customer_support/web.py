from flask import Flask, render_template, request, jsonify
import json
import re
import random
from difflib import SequenceMatcher

app = Flask(__name__)

# Load Q&A data from JSON
with open('data/qna.json', 'r') as file:
    qna_data = json.load(file)

class ChatBot:
    def __init__(self):
        self.intents = qna_data['intents']
        self.threshold = 0.6  # Similarity threshold for matching questions
    
    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        if not user_input:
            return "I didn't receive your question. Could you please type it again?"
        
        best_match = None
        highest_similarity = 0
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                similarity = SequenceMatcher(None, user_input, pattern.lower()).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = intent
        
        if best_match and highest_similarity >= self.threshold:
            return random.choice(best_match['responses'])
        
        return random.choice([
            "I'm sorry, I didn't understand that. Could you rephrase your question?",
            "I'm not sure I follow. Could you ask that differently?",
            "That's an interesting question, but I'm not programmed to answer it yet."
        ])

chatbot = ChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['message']
    bot_response = chatbot.get_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)