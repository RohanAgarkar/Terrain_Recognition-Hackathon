from flask import *
import cv2
import PIL.Image as Image
import io
import torchvision.transforms as torchv
from torchvision.transforms import Resize
import torch
import base64

from model import MyViT

class readCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def grab(self):
        ret, frame = self.cap.read()
        ret, self.buffer = cv2.imencode('.jpg', frame)
        # self.blob = (b'--frame\r\n'
            #    b'Content-Type: image/jpeg\r\n\r\n' + self.buffer.tobytes() + b'\r\n')
        self.blob = base64.b64encode(self.buffer.tobytes()).decode('utf-8')

    def getTensor(self):
        image = Image.open(io.BytesIO(self.buffer.tobytes()))
        image = torchv.Resize((256, 256))(image)
        transform = torchv.ToTensor()
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        return image

    def close(self):
        self.cap.release()

app = Flask(__name__, template_folder="template", static_folder="static")

@app.route('/')
def index():
    global cam
    cam = readCamera()
    return render_template("index.html")

@app.route('/image')
def camera():
    cam.grab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyViT((3, 256, 256), n_patches=16, n_blocks=4, hidden_d=64, n_heads=8, out_d=3).to(device)
    d = torch.load("./model.pt")["model_state_dict"]
    model.load_state_dict(d)
    pred = model(cam.getTensor())
    cls = torch.argmax(pred[0]).item()
    Text = None
    if cls == 0:
        Text = "Desert"
    elif cls == 1:
        Text = "Forest"
    elif cls == 2:
        Text = "Mountain"
    else:
        Text = ""
    response_data = {
        'text_prediction': Text,
        'image': cam.blob
    }
    return jsonify(response_data)

@app.route('/stop', methods=["GET", "POST"])
def stopcam():
    if request.method == "POST":
        data = request.json
        if data["CloseCam"] == "True":
            cam.close()
        return redirect(url_for("index"))
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
