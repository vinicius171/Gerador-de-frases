import streamlit as st
import tensorflow as tf
import numpy as np

# Carregar e preparar o texto
text = open("arquivo_formatado.txt", 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Carregar o modelo
model = tf.keras.models.load_model('modelo.h5', compile=False)

# Função para gerar texto
def generate_text(model, start_string):
    num_generate = 200
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 0.5
    
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[0]
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    text = ''.join(text_generated)
    text = text.rsplit('.', 1)[0]
    return (start_string + text + '.')

# Configurações do Streamlit
st.set_page_config(page_title='Gerador de Texto', page_icon=":eyeglasses:")
st.title('Gerador de Texto')
st.write('Inteligência Artificial treinada em obras para escrever como um autor.')

# Entrada do usuário
input_text = st.text_input('(Opcional) Escreva o início da frase:')

# Botão para gerar texto
if st.button('Gerar texto'):
    if len(input_text) == 0:
        generated_text = generate_text(model, start_string=' ')
        st.write(generated_text)
    else:
        generated_text = generate_text(model, start_string=input_text)
        st.write(generated_text)
