# Neurosity_Universal_Steam_Controller
This project uses a Neurosity Crown BCI device to translate detected mental commands (Kinesis) into keyboard presses or mouse clicks, enabling thought-based control in applications and games. It leverages a pre-trained machine learning model to interpret EEG data and uses spike detection logic to trigger actions reliably. 

The model I am providing/using is an XGBOOST model trained on the data provided in the following (https://github.com/neurosity/sw-kinesis-ai). Training and retraining scripts are available here (http://placeholder.com-volunteer-project).

I'm including the generalized model as part of this project. Perhaps through retraining, you can improve on my results. 

The script also creates a virtual Xbox 360 controller that should be automatically recognized by steam. Just map the controls in the game to script. Mapping 'push' to 'fireball' in Oblivion remastered and 'left_arm' to 'left mouse button' is a quick way to add thought commands to a AAA game. 

**MVP Goal Achieved:** Successfully maps the "Push" command spike to a 'C' key press and "Left_Arm" to a left mouse click, usable to map Kinesis Commands to the game Oblivion Remastered. Should work with any steam game. Rebind the buttons and clicks in the script however you want. 

## Features

* Connects to Neurosity Crown using the official Python SDK.
* Processes raw EEG data in real-time using custom filtering (`filterer.py`).
* Runs inference using a pre-trained XGBoost model and scaler.
* Detects "spikes" in the probability of specific mental commands.
* Triggers configurable keyboard key presses or mouse clicks via `pynput`.
* Emulates a virtual Xbox 360 controller using `vgamepad` for Steam compatibility.
* Provides real-time prediction monitoring via a WebSocket server and HTML interface.

## Prerequisites

**Hardware:**

* Neurosity Crown BCI device

**Software:**

* **Python:** Version 3.8 or higher recommended.
* **pip:** Python package installer (usually comes with Python).
* **Git:** (Optional) For cloning the repository.
* **ViGEmBus Driver:** **Required** for the `vgamepad` library to function. Download and install the latest release from [ViGEmBus Releases](https://github.com/ViGEm/ViGEmBus/releases).

## Setup Instructions

1.  **Get the Code:**
    * **Option A (Clone):** If you have Git, clone the repository:
        ```bash
        git clone <your-repository-url>
        cd <repository-directory>
        ```
    * **Option B (Download):** Download the project files (e.g., as a ZIP from GitHub) and extract them to a folder.

2.  **Create Virtual Environment (Recommended):**
    * Open a terminal or command prompt in the project directory.
    * Create a virtual environment:
        ```bash
        python -m venv .env # Creates a virtual environment named '.env'
        ```
    * Activate the virtual environment:
        * **Windows (cmd/powershell):** `.\.env\Scripts\activate`
        * **macOS/Linux (bash/zsh):** `source .env/bin/activate`
    * You should see `(.env)` appear at the beginning of your terminal prompt.

3.  **Install Dependencies:**
    * Ensure your virtual environment is active.
    * Install the required Python packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Install ViGEmBus Driver:**
    * If you haven't already, download and run the installer for the ViGEmBus driver from the link in the Prerequisites section. **This is essential for the virtual gamepad.**

5.  **Create `.env` File:**
    * In the main project directory (where the Python script is), create a file named exactly `.env` (note the leading dot).
    * Add your Neurosity credentials to this file, one per line:
        ```dotenv
        NEUROSITY_DEVICE_ID=your_crown_device_id_here
        NEUROSITY_EMAIL=your_neurosity_login_email
        NEUROSITY_PASSWORD=your_neurosity_login_password
        ```
    * Replace the placeholder values with your actual credentials.

6.  **Place Required Files:**
    * Ensure the following files are in the same directory as the main Python script (`bci_multi_action_controller.py` or similar):
        * `filterer.py` (The EEG filtering logic)
        * `kinesis_xgboost_model_softprob.json` (Your trained XGBoost model)
        * `kinesis_scaler.pkl` (Your trained scaler)
        * `.env` (The credentials file you just created)
        * `oblivion_monitor.html` or similar for monitoring.

## Configuration

Open the main Python script (`bci_multi_action_controller.py` or similar name) in a text editor:

1.  **Command Actions (`COMMAND_ACTIONS` Dictionary):**
    * This dictionary defines which mental command triggers which action.
    * Modify or add entries:
        * `"CommandName": {'type': 'key', 'value': 'k'}` for keyboard presses.
        * `"CommandName": {'type': 'mouse_click', 'value': Button.left}` 

for mouse clicks (use `Button.right`, `Button.middle` as needed). 

    * **Comment out lines** (using `#`) for commands you don't want to activate.


2.  **Spike Detection Parameters:**
    * Adjust `PROBABILITY_HISTORY_LENGTH`, `SPIKE_DELTA_THRESHOLD`, `SPIKE_ACTIVATION_THRESHOLD`, and `TRIGGER_COOLDOWN` near the top of the script to fine-tune sensitivity and prevent accidental triggers. Experimentation is key!

3. **Replace information in the .env file with your credentials (Neurosity device ID, email, and Password)

## Running the Script

1.  **Activate Virtual Environment:** If not already active, activate it (`.\.env\Scripts\activate` or `source .env/bin/activate`).
2.  **If you're having trouble with mouse clicks, try running as admin/sudo. Works fine for me as a regular user on windows 11.
3.  **Execute the Script:**
    ```bash
    python bci_multi_action_controller.py # Or your script's filename
    ```
4.  **Check Output:** Look for log messages indicating successful initialization of components (SDK login, model loading, controllers) and the start of the Flask server.

## Steam Integration

1.  **Run the Script:** Make sure the Python script is running (as Administrator).
2.  **Open Steam:** Launch Steam.
3.  **Navigate to Controller Settings:** Go to Steam -> Settings -> Controller.
4.  **Check Detection:** Steam should automatically detect a new "Xbox 360 Controller". This is the virtual gamepad created by the script.
5.  **Game Configuration:**
    * Launch your game (e.g., Oblivion).
    * The game should now receive keyboard/mouse inputs triggered by your BCI commands via the script.
    * Steam Input might try to manage the virtual controller. You may need to adjust the Steam Input configuration for the game to ensure it doesn't interfere with the direct keyboard/mouse inputs, or configure it appropriately if you plan to map BCI commands to virtual gamepad buttons in the future. For this keyboard/mouse setup, ensuring Steam Input uses a "Keyboard (WASD) and Mouse" template for the game might be simplest.

## Monitoring (Optional)

1.  While the Python script is running, open the `oblivion_monitor.html` file (or similar provided HTML file) in your web browser.
2.  It should connect to the WebSocket server run by the Python script and display real-time prediction probabilities, helping you visualize the BCI output and debug the spike detection.

