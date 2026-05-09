#relie html with python
from flask import Flask, render_template, request
from pfc import predict_msg, get_metrics  # ta fonction python


app = Flask(__name__)
history=[]

@app.route('/')
def home():
    metrics = get_metrics()  # Récupère les métriques pour les afficher sur la page d'accueil
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    result = predict_msg(message)  # appelle ton code
    metrics = get_metrics()
    history.append({
        "text": message[:60],
        "result": result
    })
    return render_template('index.html', 
                           prediction=result, 
                           metrics=metrics,
                           history=history
)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    history.clear()
    return {"success": True}

# Ouvrir navigateur automatiquement
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
