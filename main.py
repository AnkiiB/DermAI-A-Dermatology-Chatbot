from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import skin_cancer_detection as SCD

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    pic = request.files['pic']
    input_img = Image.open(pic)
    input_img = input_img.resize((28, 28))
    img = np.array(input_img).reshape(-1, 28, 28, 3)
    result = SCD.model.predict(img)
    class_id = np.argmax(result, axis=1)[0]  # Convert to int explicitly if needed
    class_probability = np.max(result) * 100  # Convert to float explicitly if needed

    # Diagnosis information and disclaimer
    classes = {
        0: ("Actinic keratosis",
            "Actinic keratosis is a rough, scaly patch on your skin that develops from years of exposure to the sun. It's most commonly found on your face, lips, ears, back of your hands, forearms, scalp or neck."),
        1: ("Basal cell carcinoma",
            "Basal cell carcinoma is a type of skin cancer that is most likely to develop on areas of skin exposed to the sun. This form of skin cancer is the least dangerous but must be treated."),
        2: ("Benign keratosis-like lesions",
            "Benign keratosis-like lesions such as seborrheic keratoses are commonly found in older adults. They appear as brown, black or light tan spots on the face, chest, shoulders or back."),
        3: ("Dermatofibroma",
            "Dermatofibromas are harmless round, reddish-brown spots caused by an overgrowth of fibrous tissue."),
        4: ("Melanocytic nevi",
            "A melanocytic nevus (also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with ‘melanocytic nevus’, but there are also sources that equate the term mole with any nevus form."),
        5: ("Pyogenic granulomas",
            "Pyogenic granulomas are small, raised, red bumps on the skin that bleed easily due to an abundance of blood vessels."),
        6: ("Melanoma",
            "Melanoma is the most dangerous form of skin cancer. If melanoma is recognized and treated early, it is almost always curable.")
    }

    class_name, description = classes.get(class_id, ("Unknown", "No description available."))
    disclaimer = "DISCLAIMER! Please consult a professional doctor for further treatment."

    # Ensure that numpy types are converted to Python native types for JSON serialization
    response = {
        'class_id': int(class_id),  # Convert to int
        'class_name': class_name,
        'class_probability': float(class_probability),  # Convert to float
        'description': description,
        'disclaimer': disclaimer
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
