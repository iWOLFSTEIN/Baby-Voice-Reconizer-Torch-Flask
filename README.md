## What it does
The program detects and classifies the baby-sound inputs and notifies the caretaker when the baby is in need. (ie. Baby Crying)

## Training
Run the following commands to run the project and train the model

- python3 -m venv env
- source env/bin/activate
- pip install -r requirements.txt
- python3 model.py

It'll generate two files of trained model, you can use any of them but recommended is to use neural_network_20.pt file

## Usage
Copy the neural_network_20.pt file to example folder, navigate to example folder using the command below in the terminal

- cd example

Now run the following commands,

- python3 -m venv env
- source env/bin/activate
- pip install -r requirements.txt
- python3 app.py

This will start a Flask server which accept an audio file and gives you analyzed results accordingly.

