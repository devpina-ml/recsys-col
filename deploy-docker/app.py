from flask import Flask, request
from coba import get_collaborative, load_model

app = Flask(__name__)

best_algo = load_model('./model.pickle')

@app.route('/api/recsys/cf')
def collaborative():
    username = request.args.get("username")
    top_n = request.args.get("top", type=int)


    return get_collaborative(algo=best_algo, username=username, top_n=top_n)

if __name__ == '__main__':
    app.run()




