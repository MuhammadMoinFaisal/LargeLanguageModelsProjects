from flask import Flask
app=Flask(__name__)
@app.route('/hello', methods=['GET', 'POST'])
def home():
    return "Hello World from Muhammad Moin"


if __name__ == '__main__':
    app.run(debug=True)