from flask import Flask, render_template, request, make_response
from generate import generate

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def create():

    if request.method == 'POST':
        if request.form.get('Generate') == 'Generate':
            path = generate()
            path = "MIDIjs.play('../" + path + "');"
            print(path)
            return render_template('index.html', prediction=path)

    else:
        return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
