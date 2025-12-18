import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope
import keras
import random

# Set page configuration
st.set_page_config(
    page_title="IntelliSort Ghana",
    page_icon="♻️",
    layout="wide"
)

# App title and description
st.title("♻️ IntelliSort: Smart Waste Classifier")
st.markdown("Take a photo of waste to classify it and get practical disposal tips")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

# Custom DepthwiseConv2D layer to handle compatibility issues
class CompatibleDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove the 'groups' parameter if it exists (for compatibility)
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Load the model with error handling and custom compatibility layer
@st.cache_resource
def load_waste_model():
    try:
        # Register custom layer to handle compatibility issues
        with custom_object_scope({'DepthwiseConv2D': CompatibleDepthwiseConv2D}):
            # Try to load the model
            model = load_model("keras_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("This is often due to version compatibility issues. Please ensure you're using the correct TensorFlow version for this model.")
        return None

# Load labels
@st.cache_resource
def load_waste_labels():
    try:
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        st.error(f"Failed to load labels: {str(e)}")
        return None

# Load model and labels
model = load_waste_model()
class_names = load_waste_labels()

# Function to get sustainability tips for Ghana
def get_sustainability_tips(waste_type):
    tips = {
        "Plastic": [
            "Rinse and dry plastic bottles/containers. These are highly valuable to collectors and recycling initiatives like Nelplast.",
            "Consider reusable alternatives: carry a reusable water bottle and shopping bag to reduce 'sachet' and 'rubber' waste.",
            "Flat plastic (like sachets) is rarely recyclable. Try to avoid it where possible and dispose of it properly to prevent clogging drains."
        ],
        "Paper": [
            "Dry paper and cardboard can be sold to scrap dealers or collected by informal waste pickers. Keep it separate and dry.",
            "Soiled or wet paper (like food packaging) cannot be recycled. Compost it if you have a garden or dispose of it with general waste.",
            "Reuse paper for packaging, notes, or crafts before finally recycling it."
        ],
        "Glass": [
            "Glass bottles are valuable! Return them to the breweries or sellers for a refund where possible.",
            "Broken glass is dangerous for waste pickers. Wrap it in paper and label it before disposal to prevent injuries.",
            "Clean, separate glass jars and bottles are often collected by informal sector recyclers."
        ],
    }
    
    for key in tips:
        if key.lower() in waste_type.lower():
            return tips[key]
    
    return [
        "The most sustainable waste is the waste that is never created. Choose products with less packaging.",
        "Proper sorting helps the informal waste sector—thousands of Ghanaians make a living from collecting recyclables.",
        "When in doubt, keep it dry and clean. Contaminated materials lose their value and can't be sold for recycling."
    ]

# Function to get random "Did You Know?" fact for Ghana
def get_did_you_know():
    facts = [
        "An estimated 90% of recycling in Ghana is carried out by the informal sector—individuals known as 'borla wura' or 'borla taxis' who collect and sell valuable materials.",
        "Accra generates over 3,000 metric tons of solid waste every day. Proper sorting at home can reduce what ends up in landfills.",
        "Plastic waste, especially from sachet water, is a major cause of drain clogging and flooding in Ghanaian cities during the rainy season.",
        "The Odaw River in Accra is one of the most polluted in the world, largely due to plastic waste and improper disposal. Proper waste sorting can help.",
        "Companies like Nelplast in Ghana recycle plastic waste into paving tiles and other construction materials, creating jobs and cleaning the environment.",
        "Burning electronic and plastic waste is common in Ghana, but it releases highly toxic chemicals (dioxins) that are dangerous to human health.",
        "Organic waste makes up about 60% of Ghana's municipal solid waste. Composting this at home could dramatically reduce landfill volumes.",
        "The Kpone Landfill site near Tema is one of Ghana's largest and is nearing capacity. Reducing and recycling waste is critical for the future.",
        "Many Ghanaian communities have no formal waste collection. Community-Based Organizations (CBOs) are often the key to managing waste effectively.",
        "‘Borla’ taxes (informal waste collection tricycles) play a vital role in waste collection in many Ghanaian neighborhoods not reached by large trucks."
    ]
    return random.choice(facts)

# Function to perform prediction
def predict_waste(image):
    if model is None or class_names is None:
        raise ValueError("Model or labels not loaded properly")
    
    # Resize and preprocess the image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1
    
    # Predict using the model
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    
    # Get class name and confidence
    class_name = class_names[index].split(' ', 1)[1] if ' ' in class_names[index] else class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# Camera interface
with col1:
    st.header("Camera Feed")
    
    # Use Streamlit's built-in camera input for simplicity
    picture = st.camera_input("Take a picture of waste")
    
    if picture:
        # Check if model and labels are loaded
        if model is None or class_names is None:
            st.error("Cannot classify image - model or labels failed to load. Please check your model files.")
        else:
            try:
                # Convert to OpenCV format
                image = Image.open(picture)
                img_array = np.array(image)
                
                # Process the image for classification
                with st.spinner("Classifying..."):
                    waste_class, confidence = predict_waste(img_array)
                
                # Display results
                with col2:
                    st.header("Results")
                    st.success(f"Class: {waste_class}")
                    st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                    
                    # Show sustainability tips
                    st.subheader("♻️ Sustainability and recycling Tips")
                    tips = get_sustainability_tips(waste_class)
                    for tip in tips:
                        st.info(tip)
                    
                    # Show "Did You Know?" section
                    st.subheader("Did You Know?")
                    fact = get_did_you_know()
                    st.success(f"{fact}")
                        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

st.markdown("---")
st.markdown("""
### About This App
This tool helps you sort your waste correctly, providing practical advice for the Ghanaian context. Proper sorting supports the informal recycling sector and helps protect our environment.

**Local Initiatives to Support:**
- **Environment 360**: Runs educational and recycling collection programs.
- **Nelplast**: Recycles plastic waste into construction materials.
- **E-Waste Ghana**: Provides safe disposal for electronic waste.

**Remember:** The best solution is always to **reduce** what you use, **reuse** what you can, and then **recycle** the rest.
""")