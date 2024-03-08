# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://media.discordapp.net/attachments/1049766372076433479/1214976643735490601/DALLE_2024-03-06_22.15.43_-_digital_art_of_human_lips_2.png?ex=65fb122d&is=65e89d2d&hm=ef28159cf21a21ee64610f8341e2f21be5a8bdab2bc07d2d5f8bb48f788592b5&=&format=webp&quality=lossless&width=498&height=498')
    st.title('LipBuddy')
    st.info('Silent Conversations, Amplified.')

st.title('LipBuddy') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'combine', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('Input Video')
        file_path = os.path.join('..','combine','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('Post processing')
        video, annotations = load_data(tf.convert_to_tensor(file_path))

#         video_np = video.numpy()
#         video_uint8 = (video_np * 255).astype(np.uint8).squeeze()

        imageio.mimsave('animation.mp4', video,fps=10)
        st.video('animation.mp4')
        
        st.info('Tokenized output')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Results')
        # Convert prediction to text
        originals = tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8')
        st.write('Original text: ',originals)

        predictions = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.write('Prediction text: ',predictions)


            
            # Tokenize the input strings into lists of words
        prediction = predictions.split()
        original = originals.split()

            # Initialize dynamic programming matrix
        dp_matrix = [[0] * (len(original) + 1) for _ in range(len(prediction) + 1)]

            # Fill the matrix
        for i in range(len(prediction) + 1):
            for j in range(len(original) + 1):
                if i == 0:
                    dp_matrix[i][j] = j
                elif j == 0:
                    dp_matrix[i][j] = i
                else:
                    substitution_cost = 0 if prediction[i - 1] == original[j - 1] else 1
                    dp_matrix[i][j] = min(
                        dp_matrix[i - 1][j] + 1,  # Deletion
                        dp_matrix[i][j - 1] + 1,  # Insertion
                        dp_matrix[i - 1][j - 1] + substitution_cost  # Substitution
                    )
            # Calculate WER
        wer = dp_matrix[len(prediction)][len(original)] / len(original)

        pred_chars = list(predictions)
        truth_chars = list(originals)
        # Initialize dynamic programming matrix
        dp_matrix = [[0] * (len(truth_chars) + 1) for _ in range(len(pred_chars) + 1)]

        # Fill the matrix
        for i in range(len(pred_chars) + 1):
            for j in range(len(truth_chars) + 1):
                if i == 0:
                    dp_matrix[i][j] = j
                elif j == 0:
                    dp_matrix[i][j] = i
                else:
                    substitution_cost = 0 if pred_chars[i - 1] == truth_chars[j - 1] else 1
                    dp_matrix[i][j] = min(
                        dp_matrix[i - 1][j] + 1,  # Deletion
                        dp_matrix[i][j - 1] + 1,  # Insertion
                        dp_matrix[i - 1][j - 1] + substitution_cost  # Substitution
                    )

            # Calculate CER
        cer = dp_matrix[len(pred_chars)][len(truth_chars)] / len(truth_chars)

        st.write('Word Error Rate',wer)
        st.write('Character Error Rate',cer)

        

