from flask import Flask

app = Flask(__name__)

running = False

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/get_status")
def get_status():
    global running
    return "running" if running else "stopped"

@app.route("/start")
def start():
    global running
    if not running:
        running = True
        return "success"
    else:
         return "error"
    
@app.route("/stop")
def stop():
    global running
    if running:
        running = False
        return "success"
    else:
         return "error"

if __name__ == '__main__':
        app.run(debug=True)