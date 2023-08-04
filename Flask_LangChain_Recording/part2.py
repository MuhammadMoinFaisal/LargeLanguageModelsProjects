from flask import Flask, request, jsonify
app=Flask(__name__)
@app.route('/hello', methods=['GET', 'POST'])
def home():
    return "Hello World from Muhammad Moin"

@app.route('/data', methods=['GET', 'POST'])
def my_data():
    if request.method=='GET':
        sample_data = {
            'message': 'Hello, Flask API --> GET METHOD',
            'data': [1,2,3,4,5]
        }
        return jsonify(sample_data)
    if request.method=='POST':
        sample_data = {
            'message': 'Hello, Flask API --> POST METHOD',
            'data': [5,6,7,8,9,10]
        }
        return jsonify(sample_data)


if __name__ == '__main__':
    app.run(debug=True)