import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask import Flask, render_template_string, request, redirect, url_for
import io
import base64


app_insight = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))

known_faces = {}

def recognize_face(embedding):
    if len(known_faces) == 0:
        return 'Unknown'
    best_match = 'Unknown'
    best_score = 0.4
    for name, known_emb in known_faces.items():
        similarity = cosine_similarity([embedding], [known_emb])[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = name
    return best_match


# Flask web app
flask_app = Flask(__name__)

HTML_TEMPLATE = '''
<html>
<head>
    <title>نظام التعرف على الوجوه</title>
    <style>
        body { font-family: Tahoma, Arial; background: #f7f7f7; margin: 0; padding: 0; }
        .container { max-width: 700px; margin: 40px auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #ccc; padding: 30px; }
        h1 { color: #2c3e50; text-align: center; }
        form { margin-bottom: 30px; }
        label { font-weight: bold; }
        input[type="text"], input[type="file"] { margin: 10px 0; }
        .result-img { max-width: 400px; border: 2px solid #2ecc71; border-radius: 8px; margin: 10px 0; }
        .btn { background: #2ecc71; color: #fff; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #27ae60; }
        .section { margin-bottom: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>نظام التعرف على الوجوه</h1>
        <div class="section">
            <h2>1. تدريب النظام (إضافة شخص جديد)</h2>
            <form method="post" enctype="multipart/form-data" action="/train">
                <label>اسم الشخص:</label><br>
                <input type="text" name="person_name" required><br>
                <label>صورة الوجه:</label><br>
                <input type="file" name="person_image" accept="image/*" required><br>
                <button class="btn" type="submit">إضافة للتدريب</button>
            </form>
        </div>
        <div class="section">
            <h2>2. اكتشاف الوجه</h2>
            <form method="post" enctype="multipart/form-data" action="/detect">
                <label>صورة للاختبار:</label><br>
                <input type="file" name="test_image" accept="image/*" required><br>
                <button class="btn" type="submit">اكتشاف</button>
            </form>
        </div>
        {% if result_img %}
        <div class="section">
            <h2>النتيجة:</h2>
            <img class="result-img" src="data:image/jpeg;base64,{{ result_img }}">
            {% if detected_names %}
                <p>الأشخاص المكتشفون: {{ detected_names }}</p>
            {% endif %}
        </div>
        {% endif %}
        <div class="section">
            <h3>الأشخاص المدربون:</h3>
            <ul>
            {% for name in known_names %}
                <li>{{ name }}</li>
            {% endfor %}
            </ul>
        </div>
    </div>
</body>
</html>
'''


@flask_app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, result_img=None, detected_names=None, known_names=list(known_faces.keys()))

@flask_app.route('/train', methods=['POST'])
def train():
    name = request.form.get('person_name')
    file = request.files.get('person_image')
    if not name or not file:
        return redirect(url_for('index'))
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    faces = app_insight.get(img)
    if len(faces) > 0:
        known_faces[name] = faces[0].embedding
    return redirect(url_for('index'))

@flask_app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('test_image')
    if not file:
        return redirect(url_for('index'))
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    faces = app_insight.get(img)
    detected_names = []
    for face in faces:
        bbox = face.bbox.astype(int)
        name = recognize_face(face.embedding)
        detected_names.append(name)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, name, (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', img_rgb)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return render_template_string(HTML_TEMPLATE, result_img=img_b64, detected_names=", ".join(detected_names), known_names=list(known_faces.keys()))

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000, debug=True)
