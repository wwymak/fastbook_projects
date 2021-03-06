import streamlit as st
from pathlib import Path
from PIL import Image as pImage
from fastai.learner import load_learner
from fastai.vision.core import PILImage

current_folder = Path(__file__).parent

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Instrument classifier')
st.markdown(f'currently supported instruments: `acoustic_guitar`,'
            f'`bass_guitar`, `drums`, `flute`, `gramophone`, `harp`, `piano`,'
            f' `saxophone`, `tabla`, `violen`')
uploader_image = st.file_uploader("upload your image to be classified", key="input_image_loader")

@st.cache(allow_output_mutation=True)
def load_model():

    return load_learner(current_folder/"instrument_classifier.pkl")


instrument_model = load_model()

if uploader_image is not None:
    image = PILImage.create(uploader_image)

    prediction, prediction_index, probability = instrument_model.predict(image)
    st.text(f"predicted: {prediction}, probability: {probability[prediction_index]}")
    st.image(image)