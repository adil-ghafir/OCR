import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
from transformers import pipeline
import pymongo


if 'final_keys' not in st.session_state:
    st.session_state.final_keys = None

# Set page configuration to widen the app
st.set_page_config(layout="wide")

# Function to validate the uploaded image
def validate_image(uploaded_image):
    try:
        # Attempt to open the image using PIL
        img = Image.open(uploaded_image)

        # Check image dimensions
        width, height = img.size
        if width <= 0 or height <= 0:
            return False, "Image dimensions should be greater than zero pixels."

        return True, "Image is valid."
    except Exception as e:
        return False, f"Error validating image: {e}"

# Function to display the image upload tab
def upload_tab():
    st.title("Upload Image")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Validate uploaded image
    if uploaded_image is not None:
        is_valid, message = validate_image(uploaded_image)

        if is_valid:
            st.success(message)

            # Store the uploaded image in session state
            st.session_state.uploaded_image = uploaded_image
        else:
            st.warning(message)
    else:
        st.warning("Please upload an image.")

# Function for image pre-processing
def preprocess_image(image):
    # Convert the UploadedFile to a PIL Image
    pil_image = Image.open(image)

    # Convert the image to grayscale
    image_gray = ImageOps.grayscale(pil_image)

    # Apply adaptive thresholding
    image_threshold = cv2.adaptiveThreshold(
        np.array(image_gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Store the thresholded image in a separate session state variable
    st.session_state.preprocessed_image = Image.fromarray(image_threshold)

    return [pil_image, image_gray, st.session_state.preprocessed_image]

# Function to display the image pre-processing tab
def preprocessing_tab():
    st.title("Image Pre-processing")

    # Access the stored uploaded image from session state
    if hasattr(st.session_state, 'uploaded_image'):
        uploaded_image = st.session_state.uploaded_image

        # Pre-process the image
        preprocessed_images = preprocess_image(uploaded_image)

        # Display original and preprocessed images
        cols = st.columns(2)

        # Display original image in the first column
        cols[0].image(preprocessed_images[0], caption="Original Image", use_column_width=True)

        # Display preprocessed images in the second column
        cols[1].image(preprocessed_images[1], caption="Grayscale Image", use_column_width=True)
        cols[1].image(preprocessed_images[2], caption="Inverted Thresholding", use_column_width=True)



def show_default_keys():
    st.title("Validate Keys")

    # Dropdown to choose the language of the invoice
    selected_language = st.selectbox("Select Language", ["English", "French", "Arabic"])

    # Set default keys based on the selected language
    default_keys = get_default_keys(selected_language)

    # Check if the preprocessed image is available
    if not hasattr(st.session_state, 'preprocessed_image') or st.session_state.preprocessed_image is None:
        st.warning("Please preprocess an image in the previous step.")
        return  # Stop execution if the preprocessed image is not available

    # Display default keys to the user in a table
    st.write("Default Keys:")
    keys_table = {"Default Keys": default_keys}
    st.table(keys_table)

    # Initialize specified_keys as an empty list
    specified_keys = []

    # Ask the user if they want to change keys
    change_keys = st.radio("Do you want to change keys?", ("No", "Yes"))

    if change_keys == "Yes":
        # If yes, show the interface to specify keys
        st.write("Specify your keys:")

        # For demonstration purposes, assume the user specifies new keys
        specified_keys = st.text_input("Enter specified keys (comma-separated)", value=", ".join(default_keys))
        specified_keys = specified_keys.split(",") if specified_keys else []

        # Display the specified keys in a table
        specified_keys_table = {"Specified Keys": specified_keys}
        st.table(specified_keys_table)

    else:
        # If no, continue with default keys
        st.write("Continue with default keys")

    # Add a "Continue" button to confirm and store the keys
    continue_button = st.button("Continue")

    # If the button is clicked, store the keys in another session state variable
    if continue_button:
        st.session_state.final_keys = specified_keys or default_keys
        # You can remove the redundant line here; it's already stored in the line above

# Add this function to get default keys based on the selected language
def get_default_keys(language):
    if language == "English":
        return [
            "Invoice Number",
            "Total Amount",
            "Due Date",
            "Vendor",
            "Billing Address",
            "Client/Company Name",
            "Purchase Order Number (P.O. Number)",
            "Invoice Date",
            "Description/Item Details",
            "Tax Information"
        ]
    elif language == "French":
        return [
            "Numéro de la facture",
            "Montant total",
            "Date d'échéance",
            "Vendeur",
            "Adresse de facturation",
            "Nom du client/de l'entreprise",
            "Numéro de commande (Numéro de bon de commande)",
            "Date de la facture",
            "Détails de l'article/description",
            "Informations fiscales"
        ]
    elif language == "Arabic":
        return [
            "رقم الفاتورة",
            "المبلغ الإجمالي",
            "تاريخ الاستحقاق",
            "البائع",
            "عنوان الفواتير",
            "اسم العميل/الشركة",
            "رقم أمر الشراء (P.O. Number)",
            "تاريخ الفاتورة",
            "تفاصيل السلعة/الوصف",
            "معلومات الضرائب"
        ]

# Continue with the rest of your code...




# Function to display the LayoutLM OCR tab
def layoutlm_ocr_tab():
    st.title("LayoutLM OCR")

    # Check if the specified keys are available
    if not hasattr(st.session_state, 'final_keys') or st.session_state.final_keys is None:
        st.warning("Please specify keys in the previous step.")
        return  # Stop execution if the specified keys are not available

    # Access the stored preprocessed image from session state
    if not hasattr(st.session_state, 'preprocessed_image') or st.session_state.preprocessed_image is None:
        st.warning("Please preprocess an image in the previous step.")
        return  # Stop execution if the preprocessed image is not available

    preprocessed_image = st.session_state.preprocessed_image

    # Display the preprocessed image
    st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

    # Create a pipeline for document question answering using LayoutLM
    pipe = pipeline("document-question-answering", model="impira/layoutlm-invoices")

    # List of keys for storing in a database
    invoice_keys = st.session_state.final_keys  # Use the keys from the previous step

    # Create questions based on keys
    questions = [f"What is the {key}?" for key in invoice_keys]

    # Button to start the OCR process
    start_button = st.button("Start OCR Process")

    # Check if the button is clicked
    if start_button:
        # Create columns for processing information and result
        col1, col2 = st.columns(2)

        # Create a row for each key and its associated spinner
        spinners = {}
        result_json = {}  # Initialize result_json outside the loop
        processing_result_placeholder = col2.subheader("Processing Result ...")
        progress_bar = col2.progress(0.0)  # Initialize progress bar

        for index, (question, key) in enumerate(zip(questions, invoice_keys)):
            # Create a spinner in the empty container
            with st.spinner(f"Processing {key}..."):
                answer = pipe(question=question, image=preprocessed_image)

            # Update the spinner text
            if answer and isinstance(answer[0], dict) and "answer" in answer[0]:
                result_json[key] = answer[0]["answer"]
                col1.write(f"*Extracted Information for {key}:* {result_json[key]}")
            else:
                result_json[key] = None
                col1.write(f"*Extracted Information for {key}:* No answer")

            # Update progress bar based on the number of answers done
            progress_value = (index + 1) / len(questions)
            progress_bar.progress(progress_value)

        # Update the "Processing Result..." message to "Result"
        processing_result_placeholder.subheader("Result")

        # Display the processing result after all keys are processed
        col2.json(result_json)
        
        # Button to save the JSON result to MongoDB
        if result_json:
            save_button = st.button("Save Result to MongoDB")
            if save_button:
                with st.spinner("Saving result to MongoDB..."):
                    # Connect to MongoDB
                    client = pymongo.MongoClient("mongodb://localhost:27017/")
                    db = client["your_database"]
                    collection = db["your_collection"]
                    
                    # Insert the JSON result into MongoDB
                    collection.insert_one(result_json)
                st.success("Result saved to MongoDB successfully")

















# Function to display the main workflow tabs
def display_workflow_tabs():
    upload, preprocessing, keys_tab, layoutlm_ocr = st.tabs(["Upload Image", "Image Pre-processing", "Keys Tab", "LayoutLM OCR"])

    # Display tabs content
    with upload:
        upload_tab()

    with preprocessing:
        preprocessing_tab()

    with keys_tab:
        show_default_keys()

    with layoutlm_ocr:
        layoutlm_ocr_tab()

# Streamlit app
def main():
    display_workflow_tabs()

if __name__ == "__main__":
    main()