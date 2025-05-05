# Filename: crown_kinesis_input_sim.py
# Description: Reads Neurosity Crown Kinesis predictions and accelerometer,
#              simulates keyboard/mouse actions based on command probability spikes,
#              (Accelerometer mouse movement disabled)
#              and broadcasts data via WebSockets.

import os
import sys
import time
import numpy as np
import xgboost as xgb
import pickle
import json
from dotenv import load_dotenv
from neurosity import NeurositySDK
import logging
from threading import Thread, Lock, Event
from flask import Flask, render_template, request
# *** WebSocket Imports ***
from flask_socketio import SocketIO, emit
from collections import deque # For efficient feature buffering
# *************************
# *** Input Simulation Import ***
try:
    from pynput import keyboard, mouse
except ImportError:
    print("ERROR: pynput library not found.")
    print("Please install it using: pip install pynput")
    sys.exit(1)
# *****************************
# *************************
import vgamepad as vg # Virtual gamepad library

# --- Assumed Filterer Import ---
# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script Directory: {script_dir}")
print(f"Current Working Directory: {os.getcwd()}")
# Explicitly add the script's directory to the Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added {script_dir} to sys.path")
print("Python sys.path:")
for path in sys.path:
    print(f"- {path}")
try:
    from filterer import Filterer # Assuming filterer.py is present
    print("Successfully imported Filterer.")
except ImportError as e:
    print(f"ERROR: Could not import Filterer. {e}")
    print("Please ensure 'filterer.py' is accessible in the Python path.")
    sys.exit(1)
# --------------------------------

# --- Configuration ---
ENV_PATH = '.env'
LOG_LEVEL = logging.INFO # DEBUG for more detail

# --- Model & Kinesis Config ---
MODEL_PATH = 'kinesis_xgboost_model_softprob.json'
SCALER_PATH = 'kinesis_scaler.pkl'
COMMAND_NAME_TO_LABEL_INDEX = { # Ensure these match your model
    "Unknown_Disappear34": 0,"Left_Foot": 1,"Left_Arm": 2,"Push": 3,
    "Tongue": 4,"Disappear22": 5,"Rest": 6,"Jumping_Jacks": 7
}
REVERSE_LABEL_MAPPING = {v: k for k, v in COMMAND_NAME_TO_LABEL_INDEX.items()}
NUM_CLASSES = len(COMMAND_NAME_TO_LABEL_INDEX)

# --- EEG Settings ---
NB_CHAN = 8 # Number of EEG channels
SFREQ = 256.0 # Sampling frequency
SIGNAL_BUFFER_LENGTH_SECS = 8 # EEG buffer length for feature calculation
SIGNAL_BUFFER_LENGTH = int(SFREQ * SIGNAL_BUFFER_LENGTH_SECS)
FILTER_LOW_HZ = 7.0 # Filter settings - adjust based on model training
FILTER_HIGH_HZ = 30.0
NEW_COV_RATE = 5 # How often to calculate features/predict (samples)

# --- Input Simulation Settings ---
# Kinesis Actions
ACTION_THRESHOLD = 0.70  # Probability needed to trigger action (adjust this!)
RELEASE_THRESHOLD = 0.50 # Probability needed to release action (lower than trigger)
# # Accelerometer Mouse Movement *** DISABLED ***
# MOUSE_SENSITIVITY = 1.5
# MOUSE_DEADZONE_ANGLE = 3.0
# MOUSE_MAX_ANGLE = 25.0
# INVERT_MOUSE_PITCH = True
# INVERT_MOUSE_ROLL = False

SIMULATION_INTERVAL = 1 / 30.0 # How often to check inputs (seconds, ~30Hz)

# --- Logging Setup ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s %(message)s")
logger = logging.getLogger(__name__)
if logger.hasHandlers(): logger.handlers.clear()
logger.setLevel(LOG_LEVEL)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Flask & SocketIO App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Alteredforgithub' # CHANGE THIS!
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="127.0.0.1:5000")
logging.info("Flask-SocketIO initialized.")

# --- Global Variables & Locks ---
neurosity = None
model = None
scaler = None
filterer = None
# Pynput controllers
keyboard_controller = None
mouse_controller = None

# Neurosity subscriptions
raw_unsubscribe = None
predictions_unsubscribe = None
accelerometer_unsubscribe = None # Keep subscription for potential WebSocket use

# Threading Locks
data_processing_lock = Lock()
control_lock = Lock() # Lock for prediction data, accel data, and action states

# Data storage
cov_counter = 0
latest_prediction = {"probabilities": None, "predicted_label": "Initializing...", "timestamp": 0}
latest_kinesis = {"label": "Initializing...", "confidence": 0.0, "timestamp": 0} # Still store direct Kinesis if needed
latest_accel = {"pitch": 0.0, "roll": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "timestamp": 0}

# Action states (to prevent continuous triggering)
push_action_state = False
left_arm_action_state = False
tongue_action_state = False

# --- Helper Functions --- (load_dotenv_config, connect_and_login, load_model_and_scaler, initialize_filterer)
# (No changes needed in these)
def load_dotenv_config():
    logging.info(f"Loading environment variables from: {ENV_PATH}")
    if not os.path.exists(ENV_PATH):
        logging.error(f".env file missing at {ENV_PATH}.")
        return False
    load_dotenv(dotenv_path=ENV_PATH)
    logging.info("Environment variables loaded.")
    return True

def connect_and_login():
    global neurosity
    logging.info("--- Attempting Neurosity Connection and Login ---")
    device_id = os.getenv("NEUROSITY_DEVICE_ID")
    email = os.getenv("NEUROSITY_EMAIL")
    password = os.getenv("NEUROSITY_PASSWORD")
    if not all([device_id, email, password]):
        logging.error("CONNECTION FAILED: Missing Neurosity credentials.")
        return False
    try:
        neurosity = NeurositySDK({"device_id": device_id})
        logging.info("SDK Initialized.")
        neurosity.login({"email": email, "password": password})
        logging.info("Login request sent. Waiting...")
        time.sleep(5) # Simple wait
        logging.info(">>> Neurosity Login SUCCEEDED (presumably). <<<")
        return True
    except Exception as e:
        logging.error(f">>> LOGIN FAILED: {e} <<<", exc_info=True)
        neurosity = None
        return False

def load_model_and_scaler():
    global model, scaler
    model_loaded = False
    scaler_loaded = False
    logging.info(f"Attempting to load XGBoost model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logging.warning(f"Model NOT FOUND: {MODEL_PATH}")
        model = None
    else:
        try:
            model = xgb.Booster()
            model.load_model(MODEL_PATH)
            logging.info("XGBoost model loaded.")
            model_loaded = True
        except Exception as e:
            logging.error(f"FAILED to load model: {e}", exc_info=True)
            model = None

    logging.info(f"Attempting to load scaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH):
        logging.warning(f"Scaler NOT FOUND: {SCALER_PATH}")
        scaler = None
    else:
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logging.info("Scaler loaded.")
            scaler_loaded = True
        except Exception as e:
            logging.error(f"FAILED to load scaler: {e}", exc_info=True)
            scaler = None

    if not model_loaded or not scaler_loaded:
        logging.warning("Prediction DISABLED (model/scaler load failed).")
        return False
    else:
        logging.info("Prediction ENABLED.")
        return True

def initialize_filterer():
    global filterer
    logging.info("Initializing Filterer...")
    filterer_defined = 'Filterer' in globals() or 'Filterer' in locals()
    logging.info(f"Checking Filterer before init: Defined = {filterer_defined}")
    if not filterer_defined:
         logging.error("Filterer class not defined before initialization attempt.")
         return False
    try:
        filterer = Filterer(filter_high=FILTER_HIGH_HZ, filter_low=FILTER_LOW_HZ, nb_chan=NB_CHAN, sample_rate=SFREQ, signal_buffer_length=SIGNAL_BUFFER_LENGTH)
        logging.info(f"Filterer initialized (Buffer: {SIGNAL_BUFFER_LENGTH} samples, Rate: {SFREQ}Hz, Filter: {FILTER_LOW_HZ}-{FILTER_HIGH_HZ}Hz)")
        return True
    except NameError as ne:
        logging.error(f"NameError during Filterer initialization: {ne}. Check EEG Settings constants.", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Failed to initialize Filterer: {e}", exc_info=True)
        return False

# --- Neurosity Callbacks --- (handle_raw_eeg, handle_kinesis, handle_accelerometer)
# (No changes needed in these callback function bodies)
def handle_raw_eeg(brainwave_data):
    """Processes raw EEG, generates features, predicts, and stores result."""
    global cov_counter, filterer, model, scaler, latest_prediction
    if filterer is None or model is None or scaler is None:
        return

    with data_processing_lock:
        try:
            raw_data = brainwave_data.get('data')
            info = brainwave_data.get('info', {})
            start_time = info.get('startTime', time.time() * 1000)
            num_samples = len(raw_data[0]) if raw_data and len(raw_data) > 0 else 0
            if not raw_data or num_samples == 0: return

            parsed_array = np.zeros((10, num_samples))
            for i in range(NB_CHAN):
                if i < len(raw_data): parsed_array[i, :] = raw_data[i]
            timestamps = np.linspace(start_time, start_time + (num_samples - 1) * (1000 / SFREQ), num_samples)
            parsed_array[9, :] = timestamps

            filterer.partial_transform(parsed_array)
            cov_counter += num_samples

            if cov_counter >= NEW_COV_RATE:
                current_timestamp_ms = int(time.time() * 1000)
                cov_counter = 0
                cov_matrix = filterer.get_cov()

                if cov_matrix is None or cov_matrix.size == 0: return

                features_flat = cov_matrix.flatten().reshape(1, -1)
                features_scaled = None
                if features_flat.shape == (1, 64):
                    try:
                        features_scaled = scaler.transform(features_flat)
                    except Exception as scale_e:
                        logging.error(f"Scaler error: {scale_e}")
                        return
                else:
                    logging.warning(f"Feature shape mismatch: Expected (1, 64), got {features_flat.shape}")
                    return

                if features_scaled is not None:
                    probabilities_dict = None
                    top_predicted_label = "Prediction Error"
                    exec_time_ms = 0
                    probs = None
                    try:
                        pred_start_time = time.time()
                        dmatrix = xgb.DMatrix(features_scaled)
                        probabilities_array = model.predict(dmatrix)
                        exec_time_ms = (time.time() - pred_start_time) * 1000

                        if probabilities_array is not None and probabilities_array.shape[1] == NUM_CLASSES:
                            probs = probabilities_array[0]
                            probabilities_dict = {REVERSE_LABEL_MAPPING.get(i, f"Unk_{i}"): round(float(p), 3) for i, p in enumerate(probs)}
                            top_predicted_index = np.argmax(probs)
                            top_predicted_label = REVERSE_LABEL_MAPPING.get(top_predicted_index, f"Unknown_{top_predicted_index}")
                        else:
                            top_predicted_label = "Prediction Invalid Result"

                    except Exception as pred_e:
                        logging.error(f"Error during prediction: {pred_e}", exc_info=True)
                        top_predicted_label = "Prediction Exception"
                    finally:
                        new_prediction = {
                                "probabilities": probabilities_dict,
                                "predicted_label": top_predicted_label,
                                "executionTime": round(exec_time_ms, 2),
                                "timestamp": current_timestamp_ms
                            }
                        with control_lock:
                            latest_prediction = new_prediction
                        socketio.emit('prediction_update', new_prediction)

        except Exception as e:
            logging.error(f"Error in handle_raw_eeg callback: {e}", exc_info=True)


def handle_kinesis(kinesis_data): # Handles prediction stream
    """Stores latest Kinesis prediction and emits via WebSocket."""
    global latest_kinesis
    try:
        label = kinesis_data.get("label", "N/A")
        confidence = kinesis_data.get("confidence", 0.0)
        timestamp = kinesis_data.get("timestamp", int(time.time() * 1000))
        kinesis_info = {
            "label": label,
            "confidence": round(confidence, 3),
            "timestamp": timestamp
        }
        with control_lock:
            latest_kinesis = kinesis_info
        socketio.emit('kinesis_update', kinesis_info)
        logging.debug(f"Prediction Stream Update: {label} ({confidence:.2f})")
    except Exception as e:
        logging.error(f"Error processing/emitting prediction stream data: {e}", exc_info=True)


def handle_accelerometer(accel_data):
    global latest_accel
    try:
        timestamp = accel_data.get("timestamp", int(time.time() * 1000))
        data_to_store = {
            key: round(accel_data.get(key, 0.0), 3)
            for key in ["pitch", "roll", "x", "y", "z"]
        }
        data_to_store["timestamp"] = timestamp

        with control_lock:
            latest_accel = data_to_store
        socketio.emit('accelerometer_update', latest_accel)
    except Exception as e:
        logging.error(f"Error processing/emitting accelerometer data: {e}", exc_info=True)


# --- Input Simulation Thread --- UPDATED
def input_simulator():
    """Periodically checks Kinesis predictions & accelerometer, simulates keyboard/mouse."""
    global latest_prediction, latest_accel, keyboard_controller, mouse_controller
    global push_action_state, left_arm_action_state, tongue_action_state

    logging.info("Input simulator thread started.")
    last_mouse_move_time = time.time()

    while True:
        if keyboard_controller is None or mouse_controller is None:
            logging.warning("Input controllers not ready, skipping simulation cycle.")
            time.sleep(0.5)
            continue

        # Get the latest data safely
        with control_lock:
            current_prediction = json.loads(json.dumps(latest_prediction)) # Deep copy
            # current_accel = latest_accel.copy() # We don't need accel data for this version

        # --- Kinesis Action Simulation ---
        if current_prediction and current_prediction.get("probabilities"):
            probs = current_prediction["probabilities"]

            # Push -> Left Mouse Click
            prob_push = probs.get("Push", 0.0)
            try:
                if prob_push > ACTION_THRESHOLD and not push_action_state:
                    mouse_controller.press(mouse.Button.left)
                    push_action_state = True
                    logging.info(f"ACTION: Push detected (Prob: {prob_push:.2f}) -> Left Mouse Pressed")
                elif prob_push < RELEASE_THRESHOLD and push_action_state:
                    mouse_controller.release(mouse.Button.left)
                    push_action_state = False
                    logging.info(f"ACTION: Push dropped (Prob: {prob_push:.2f}) -> Left Mouse Released")
            except Exception as e:
                logging.error(f"Error simulating left click: {e}")
                push_action_state = False # Reset state on error

            # Left Arm -> 'C' Key Press
            prob_left_arm = probs.get("Left_Arm", 0.0)
            try:
                if prob_left_arm > ACTION_THRESHOLD and not left_arm_action_state:
                    keyboard_controller.press('c')
                    left_arm_action_state = True
                    logging.info(f"ACTION: Left Arm detected (Prob: {prob_left_arm:.2f}) -> 'c' Key Pressed")
                elif prob_left_arm < RELEASE_THRESHOLD and left_arm_action_state:
                    keyboard_controller.release('c')
                    left_arm_action_state = False
                    logging.info(f"ACTION: Left Arm dropped (Prob: {prob_left_arm:.2f}) -> 'c' Key Released")
            except Exception as e:
                logging.error(f"Error simulating 'c' key: {e}")
                left_arm_action_state = False # Reset state on error


            # Tongue -> Right Mouse Click
            prob_tongue = probs.get("Tongue", 0.0)
            try:
                if prob_tongue > ACTION_THRESHOLD and not tongue_action_state:
                    mouse_controller.press(mouse.Button.right)
                    tongue_action_state = True
                    logging.info(f"ACTION: Tongue detected (Prob: {prob_tongue:.2f}) -> Right Mouse Pressed")
                elif prob_tongue < RELEASE_THRESHOLD and tongue_action_state:
                    mouse_controller.release(mouse.Button.right)
                    tongue_action_state = False
                    logging.info(f"ACTION: Tongue dropped (Prob: {prob_tongue:.2f}) -> Right Mouse Released")
            except Exception as e:
                logging.error(f"Error simulating right click: {e}")
                tongue_action_state = False # Reset state on error

        # --- Accelerometer Mouse Movement Simulation --- *** COMMENTED OUT ***
        # if current_accel:
        #     pitch = current_accel.get("pitch", 0.0)
        #     roll = current_accel.get("roll", 0.0)
        #
        #     # Calculate scaled movement deltas
        #     mouse_dx = 0
        #     mouse_dy = 0
        #
        #     # Roll controls horizontal (X) movement
        #     if abs(roll) > MOUSE_DEADZONE_ANGLE:
        #         # Clamp angle between -max_angle and +max_angle
        #         clamped_roll = max(-MOUSE_MAX_ANGLE, min(MOUSE_MAX_ANGLE, roll))
        #         # Normalize to -1.0 to 1.0 range
        #         scaled_roll = clamped_roll / MOUSE_MAX_ANGLE
        #         # Calculate delta movement (adjust multiplier for desired speed)
        #         mouse_dx = int(scaled_roll * MOUSE_SENSITIVITY * 10)
        #         if INVERT_MOUSE_ROLL:
        #             mouse_dx = -mouse_dx
        #
        #     # Pitch controls vertical (Y) movement
        #     if abs(pitch) > MOUSE_DEADZONE_ANGLE:
        #         # Clamp angle
        #         clamped_pitch = max(-MOUSE_MAX_ANGLE, min(MOUSE_MAX_ANGLE, pitch))
        #         # Normalize
        #         scaled_pitch = clamped_pitch / MOUSE_MAX_ANGLE
        #         # Calculate delta movement
        #         mouse_dy = int(scaled_pitch * MOUSE_SENSITIVITY * 10)
        #         if INVERT_MOUSE_PITCH:
        #             mouse_dy = -mouse_dy # Invert Y-axis for typical FPS look
        #
        #     # Move the mouse if there's any delta
        #     if mouse_dx != 0 or mouse_dy != 0:
        #         try:
        #             # pynput's move is relative
        #             mouse_controller.move(mouse_dx, mouse_dy)
        #             # logging.debug(f"Mouse Moved: dx={mouse_dx}, dy={mouse_dy}") # Very verbose
        #         except Exception as e:
        #             logging.error(f"Error moving mouse: {e}")
        # --- End Accelerometer Mouse Movement ---

        # Sleep for the defined interval
        time.sleep(SIMULATION_INTERVAL)


# --- Flask Routes --- (Same as before)
@app.route('/')
def status_page():
    return "BCI Input Simulator Bridge Running. Connect via WebSocket."

# --- WebSocket Event Handlers --- (Same as before)
@socketio.on('connect')
def handle_connect():
    client_id = request.sid if request else 'Unknown'
    logging.info(f"WebSocket Client connected: {client_id}")
    emit('connection_ack', {'message': 'BCI Bridge connected!'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid if request else 'Unknown'
    logging.info(f"WebSocket Client disconnected: {client_id}")

# --- Neurosity Streaming Function ---
def neurosity_stream_runner():
    """Subscribes to required Neurosity streams."""
    global raw_unsubscribe, predictions_unsubscribe, accelerometer_unsubscribe, neurosity
    if not neurosity: logging.error("Neurosity object missing."); return

    logging.info("Starting Neurosity stream runner thread...")
    while True:
        subscriptions = {"raw": False, "predictions": False, "accel": False}
        local_raw_unsub, local_preds_unsub, local_accel_unsub = None, None, None
        connected_and_subscribed = False

        try:
            # --- Attempt Subscriptions ---
            try:
                logging.info("Attempting to subscribe to raw brainwaves...")
                if model and scaler:
                    local_raw_unsub = neurosity.brainwaves_raw(handle_raw_eeg)
                    raw_unsubscribe = local_raw_unsub
                    subscriptions["raw"] = True
                    logging.info("Raw EEG subscribed.")
                else:
                    logging.warning("Raw EEG subscription skipped (model/scaler not loaded).")
            except Exception as e: logging.error(f"Failed raw EEG sub: {e}", exc_info=True)

            try:
                logging.info("Attempting to subscribe to Predictions...")
                local_preds_unsub = neurosity.predictions(handle_kinesis)
                predictions_unsubscribe = local_preds_unsub
                subscriptions["predictions"] = True
                logging.info("Predictions subscribed.")
            except Exception as e: logging.error(f"Failed Predictions sub: {e}", exc_info=True)

            try:
                logging.info("Attempting to subscribe to accelerometer...")
                local_accel_unsub = neurosity.accelerometer(handle_accelerometer)
                accelerometer_unsubscribe = local_accel_unsub
                subscriptions["accel"] = True
                logging.info("Accelerometer subscribed.")
            except AttributeError: logging.warning("'.accelerometer()' method missing or failed. Accel data unavailable.")
            except Exception as e: logging.error(f"Failed accel sub: {e}", exc_info=True)

            # Check if essential subscriptions succeeded
            essential_subs_ok = subscriptions["predictions"] or (subscriptions["raw"] and model and scaler)
            # Accel is optional now for core functionality
            if not essential_subs_ok:
                logging.error("Failed essential subscriptions (Predictions/Raw). Retrying in 15s.")
                time.sleep(15)
                continue

            logging.info(">>> Waiting for Neurosity data... <<<")
            connected_and_subscribed = True
            while True: time.sleep(60)

        except Exception as e:
            logging.error(f"Stream runner error: {e}. Retrying in 15s.", exc_info=True)

        finally:
            logging.info("Stream runner cleanup/pause...")
            if local_raw_unsub:
                try: local_raw_unsub(); logging.debug("Attempted raw unsubscribe.")
                except Exception as unsub_e: logging.error(f"Error during raw unsubscribe: {unsub_e}")
            if local_preds_unsub:
                try: local_preds_unsub(); logging.debug("Attempted predictions unsubscribe.")
                except Exception as unsub_e: logging.error(f"Error during predictions unsubscribe: {unsub_e}")
            if local_accel_unsub:
                try: local_accel_unsub(); logging.debug("Attempted accel unsubscribe.")
                except Exception as unsub_e: logging.error(f"Error during accel unsubscribe: {unsub_e}")

            raw_unsubscribe = None
            predictions_unsubscribe = None
            accelerometer_unsubscribe = None

            logging.info("Waiting 15s before reconnect attempt...")
            time.sleep(15)


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Neurosity Kinesis -> Keyboard/Mouse Input Simulator + WebSocket Bridge ---")
    if not load_dotenv_config(): sys.exit(1)

    # Initialize BCI components
    if not initialize_filterer(): sys.exit(1)
    if not load_model_and_scaler():
        print("WARNING: Model/Scaler failed to load. Predictions will be disabled.")

    if not connect_and_login(): sys.exit(1)

    # Create pynput controllers
    try:
        keyboard_controller = keyboard.Controller()
        mouse_controller = mouse.Controller()
        logging.info("Pynput keyboard and mouse controllers created.")
    except Exception as e:
        logging.error(f"Failed to create pynput controllers: {e}", exc_info=True)
        if neurosity and hasattr(neurosity, 'disconnect'): neurosity.disconnect()
        sys.exit(1)

    # Start Neurosity streaming thread
    neurosity_thread = Thread(target=neurosity_stream_runner, name="NeurosityStreamThread", daemon=True)
    neurosity_thread.start()

    # Start Input Simulation thread
    input_thread = Thread(target=input_simulator, name="InputSimulatorThread", daemon=True)
    input_thread.start()

    # Start Flask-SocketIO server
    logging.info("Starting Flask-SocketIO server...")
    print("-----------------------------------------------------")
    print(f">>> Using Kinesis Model: {MODEL_PATH} <<<")
    print(">>> Simulating Keyboard/Mouse based on Kinesis Commands <<<") # Updated desc
    print(f">>> Push -> Left Click | Left Arm -> 'c' | Tongue -> Right Click (Threshold: {ACTION_THRESHOLD:.2f}) <<<")
    print(f">>> Accel Pitch/Roll -> Mouse Movement (DISABLED) <<<") # Updated mouse info
    print(">>> Broadcasting data via WebSocket on ws://127.0.0.1:5000 <<<")
    print("-----------------------------------------------------")

    try:
        socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    except OSError as e:
        logging.error(f"Flask server failed (OSError): {e}")
        print(f"\nERROR: Port 5000 likely in use.\n")
    except Exception as e:
        logging.error(f"Flask server failed (Exception): {e}", exc_info=True)
    finally:
        # Shutdown sequence
        logging.info("Shutdown sequence initiated...")
        print("\nShutting down... please wait.")

        # Attempt final unsubscribes
        logging.info("Attempting final stream unsubscribes...")
        if raw_unsubscribe:
            try: raw_unsubscribe(); logging.info("Final raw unsubscribe called.")
            except Exception as e: logging.error(f"Error in final raw unsubscribe: {e}")
        if predictions_unsubscribe:
            try: predictions_unsubscribe(); logging.info("Final predictions unsubscribe called.")
            except Exception as e: logging.error(f"Error in final predictions unsubscribe: {e}")
        if accelerometer_unsubscribe:
            try: accelerometer_unsubscribe(); logging.info("Final accelerometer unsubscribe called.")
            except Exception as e: logging.error(f"Error in final accel unsubscribe: {e}")

        # Release any held keys/buttons on exit
        logging.info("Releasing any potentially held keys/buttons...")
        try:
            if push_action_state: mouse_controller.release(mouse.Button.left)
            if left_arm_action_state: keyboard_controller.release('c')
            if tongue_action_state: mouse_controller.release(mouse.Button.right)
        except Exception as e:
            logging.error(f"Error releasing inputs during shutdown: {e}")

        if neurosity and hasattr(neurosity, 'disconnect'):
             try:
                 logging.info("Disconnecting SDK..."); neurosity.disconnect(); logging.info("SDK Disconnected.")
             except Exception as disc_e:
                 logging.error(f"Error during SDK disconnect: {disc_e}")

        logging.info("Shutdown complete.")
        print("Shutdown complete.")
