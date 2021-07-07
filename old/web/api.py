import flask
from flask import request, jsonify, redirect, url_for
from random import random, randint, sample
import json


app = flask.Flask(__name__)
app.config["DEBUG"] = True

COUNTRIES = ['BLR', 'DN', 'FI', 'NO', 'RUS', 'SW', 'UA', 'UK', 'USA']


action = {
    'campaign_id': 1283,
    'ad_id': 283,
    'segment': 'poor',
    'build_id': sample(['A', 'B'], 1)[0],  # A or B
    'items': {
        'time_spent': 324,  # in sec
        'time_hover': 21,
        'time_session': 1894
    },
    'user': {
        'ip': '{}.{}.{}.{}'.format(*sample(range(0,255),4)),
        'age': randint(14, 60),
        'country': sample(COUNTRIES, 1)[0]
    }
}


@app.route('/', methods=['GET'])
def index():
    if random() > 0.5:
        return "<div style='padding: 200px;'><a href='/exit' style='display: block; margin: 0 auto; width: 150px; text-align: center; text-decoration: none; background: red; color: white; padding: 10px;'>Red button</a></div>"
    else:
        return "<div style='padding: 200px;'><a href='/exit' style='display: block; margin: 0 auto; width: 150px; text-align: center; text-decoration: none; background: green; color: white; padding: 10px;'>Green button</a></div>"

# @app.route('/send', methods=['GET'])
# def send():
#     return jsonify(action)

@app.route('/exit', methods=['GET'])
def exit():
    file = open('../data/actions.json', 'r+', encoding='utf-8')
    acts = file.read()
    actions = [] if acts == '' else [json.loads(acts)]
    actions.append(action)
    file.truncate(0)
    json.dump(actions, file, indent=2)
    file.close()

    return '//{}//'.format(str(actions))
    # return "<h1>Nicely done! Your action has been recorded!</h1><br>" \
    #        "<a href='/'>Go home</a>"


# API
# A route to return all of the available entries in our catalog.
@app.route('/api/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(actions)

@app.route('/api/resources/books', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for book in actions:
        if book['id'] == id:
            results.append(book)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

app.run()