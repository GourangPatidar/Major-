import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = MobileNetV2(weights='imagenet')

# Expanded mapping
food_calorie_dict = {
    'cheeseburger': 300,
    'pizza': 285,
    'hotdog': 150,
    'ice_cream': 210,
    'salad': 120,
    'banana': 90,
    'apple': 95,
    'granny_smith': 95,
    'orange': 62,
    'carrot': 41,
    'broccoli': 55,
    'strawberry': 4,
    'sandwich': 250,
    'french_loaf': 270,
    'bagel': 250,
    'mushroom': 22,
}

# Prediction function
def predict_food_and_calories(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]

    for pred in decoded_preds:
        food_name = pred[1].lower()

        # Direct match
        if food_name in food_calorie_dict:
            calories = food_calorie_dict[food_name]
            return food_name, calories, pred[2]*100

        # If contains fruit, apple, etc.
        if "apple" in food_name or "fruit" in food_name:
            return "apple", 95, pred[2]*100

    # If no match
    top_pred_name = decoded_preds[0][1]
    top_pred_conf = decoded_preds[0][2] * 100
    return None, None, (top_pred_name, top_pred_conf)

# Streamlit app
st.title("🍔 Food Calorie Predictor")
st.write("Upload a food image and get estimated calories!")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Predicting...'):
        food_name, calories, info = predict_food_and_calories(img)

    if food_name:
        st.success(f"🍴 Detected Food: **{food_name.replace('_', ' ').capitalize()}**")
        st.info(f"🔥 Estimated Calories: **{calories} kcal**")
    else:
        st.warning(f"🤔 Best Guess: **{info[0].replace('_', ' ').capitalize()}** with {info[1]:.2f}% confidence")
        st.error("😔 Couldn't recognize the food properly or not in database.")
