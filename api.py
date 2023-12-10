import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import easyocr
from openai import OpenAI
import tiktoken
import requests
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app)

# Configuration for file uploads
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
configure_uploads(app, photos)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "photo" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["photo"]

    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = photos.save(file)
        result = ocr(filename)
        return result
    else:
        return jsonify({"message": "File type not allowed"}), 400


def ocr(filename):
    model = torch.load('models/findbox.pth')
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    original_img = Image.open('uploads/' + filename).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(original_img).unsqueeze(0).to(device)

    # apply uploaded image into the boundary prediction model

    output = model(img)
    boxes = output[0]['boxes'].detach().to(device)
    scores = output[0]['scores'].detach().to(device)

    # find the box with the highest condifence score
    max_score_idx = torch.argmax(scores)
    boxes = boxes[max_score_idx].unsqueeze(0)
    scores = scores[max_score_idx].unsqueeze(0)

    # Convert image into Tensor of shape (C x H x W) and dtype uint8
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img_tensor = transform(original_img)

    # draw boundary on the image
    toPilTransform = transforms.Compose([
        transforms.ToPILImage()
    ])

    output_image = toPilTransform(draw_bounding_boxes(
        image=img_tensor, boxes=boxes, colors=(255, 0, 0), width=3))

    # convert image into base64 string
    image_buffer = BytesIO()
    output_image.save(image_buffer, format="PNG")
    base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    # crop the image based on the box predicted

    img = Image.open('uploads/' + filename).convert("L")
    box = tuple(boxes[0].tolist())
    img = img.crop(box)
    img.save('uploads/' + filename[:-4] + '_cropped.png')

    # perform ocr on cropped section

    reader = easyocr.Reader(['en'])
    text = reader.readtext('uploads/' + filename[:-4] + '_cropped.png')
    text_list = []
    for t in text:
        t = (t[1], round(t[2],2))
        text_list.append(t)
    # print(text_list)
    try:
        parseWithGPT(text_list)
    except:
        print("chat gpt not working")  
    result = {
        "image": base64_image,
        "text": text_list,
        "score": scores.item()
    }
    return result

def parseWithGPT(text_list):
    prompt = 'Extract the full name and the address from this easyocr result.\n ' + ', '.join([f"('{item[0]}', {item[1]})" for item in text_list])
    data = {
        'content': prompt,
        'role': "user"
    }
    print(num_tokens_from_messages([data]))
    chatgpt = OpenAI(api_key = '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')
    response = chatgpt.chat.completions.create(
        model='gpt-3.5-turbo-0613',
        response_format={ "type": "json_object" },
        messages=[data],
        temperature=0
    )
    print(response)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "gif"}


if __name__ == "__main__":
    app.run(port=8080, debug=True)
