from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this for security

# ✅ Model setup
MODEL_PATH = "DFU_Model_FIXED_REBUILT.keras"
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['Diseased', 'Normal']
preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

# ✅ File paths
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

# ---------------------------------------------
# 🔹 Helper functions
# ---------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# ---------------------------------------------
# 🔹 Routes
# ---------------------------------------------
@app.route('/')
def index():
    return render_template('login.html')

# ✅ Registration
@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    users = load_json(USERS_FILE)

    if username in users:
        return render_template('login.html', error="⚠️ Username already exists.")

    users[username] = {"password": password, "role": "user"}
    save_json(USERS_FILE, users)
    return render_template('login.html', message="✅ Registration successful! Please log in.")

# ✅ Login
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']  # user/admin
    users = load_json(USERS_FILE)

    if username not in users or users[username]['password'] != password:
        return render_template('login.html', error="❌ Invalid credentials.")

    if role == 'admin' and users[username]['role'] != 'admin':
        return render_template('login.html', error="🚫 Not authorized as admin.")

    session['username'] = username
    session['role'] = users[username]['role']

    if session['role'] == 'admin':
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('user_dashboard'))

# ✅ Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ---------------------------------------------
# 🔹 User Dashboard
# ---------------------------------------------
@app.route('/user')
def user_dashboard():
    if 'username' not in session or session['role'] != 'user':
        return redirect(url_for('index'))
    return render_template('user_dashboard.html', username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('index'))

    if 'file' not in request.files:
        return "⚠️ No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "⚠️ Empty filename", 400

    # Save temporarily
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Preprocess image
    img = Image.open(file_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_fn(img_array)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "Diseased 🩸" if prediction < 0.5 else "Normal ✅"

    # Save history
    history = load_json(HISTORY_FILE)
    username = session['username']
    if username not in history:
        history[username] = []

    history[username].append({
        "filename": file.filename,
        "result": result,
        "confidence": float(prediction),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(HISTORY_FILE, history)

    return render_template(
        'result.html',
        result=result,
        confidence=float(prediction),
        image_path=file_path
    )

# ---------------------------------------------
# 🔹 Admin Dashboard
# ---------------------------------------------
@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('index'))
    history = load_json(HISTORY_FILE)
    return render_template('admin_dashboard.html', history=history)

@app.route('/delete_history/<username>', methods=['POST'])
def delete_history(username):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('index'))

    history = load_json(HISTORY_FILE)
    if username in history:
        del history[username]
        save_json(HISTORY_FILE, history)
    return redirect(url_for('admin_dashboard'))

# ---------------------------------------------
# 🔹 Static Pages
# ---------------------------------------------
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ---------------------------------------------
# 🔹 Run App
# ---------------------------------------------
if __name__ == '__main__':
    # Create admin if not present
    users = load_json(USERS_FILE)
    if "admin" not in users:
        users["admin"] = {"password": "admin123", "role": "admin"}
        save_json(USERS_FILE, users)

    app.run(debug=True)