from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def data():
    if request.method == 'GET':
        print('got GET...')
        return render_template('index.html')
    if request.method == 'POST':
        input_text = request.form['article']
        generated_text = input_text
        return render_template('index.html', generated_text=generated_text)


if __name__ == '__main__':
    app.run(debug=True)
