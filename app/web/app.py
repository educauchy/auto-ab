from sqlalchemy import create_engine
from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy


# engine = create_engine('dialect+driver://username:password@host:port/database')
# engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')
app = Flask(__name__, static_url_path='/static')
app.config.from_object("config.Config")
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)

    def __init__(self, email):
        self.email = email

@app.route('/')
def hello_world():
    return jsonify(hello="world")
    # img_path = 'sim_plot.png'
    # return render_template('index.html', img_path=img_path)

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5050, debug=True)