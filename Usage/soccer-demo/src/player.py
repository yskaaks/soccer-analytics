import streamlit as st
import cv2
import numpy as np
import socket
import struct
import pickle
import time


global first_attempt
first_attempt = True
def attempt_connection(host, port, status_placeholder):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        status_placeholder.empty()  # Clear status message upon successful connection
        return client_socket
    except ConnectionRefusedError:
        status_placeholder.text("Server is not available. Retrying...")
        time.sleep(2)
        return None

def receive_and_display_streams(host='127.0.0.1', port=8080):
    st.title("Soccer Analysis")

    status_placeholder = st.empty()  # Placeholder for showing connection status

    client_socket = None
    while True:
        if client_socket is None:
            client_socket = attempt_connection(host, port, status_placeholder)
            if client_socket is None:
                continue

        data = b""
        payload_size = struct.calcsize("Q")
        global first_attempt
        if first_attempt:
            cols = st.columns(2)
            placeholders = [cols[i % 2].empty() for i in range(4)]
            first_attempt = False
        captions = ['Frame', 'Layout', 'Heatmap 1', 'Heatmap 2']

        try:
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:  # Connection closed by the server
                        raise Exception("Connection closed")
                    data += packet

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frames = pickle.loads(frame_data)
                for i, frame in enumerate(frames):
                    if len(frame.shape) == 2:  # Convert grayscale images for display
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    placeholders[i].image(frame, caption=captions[i], use_column_width=True)

        except Exception as e:
            # st.error("Connection lost. Trying to reconnect...")
            # placeholders = [cols[i % 2].empty() for i in range(4)]
            # status_placeholder = st.empty()
            client_socket.close()
            client_socket = None

if __name__ == '__main__':
    receive_and_display_streams()
