# DoC_ML_CW1

### Running Code on Lab Machines:

- Download the source code onto the Lab Machine.
- Extract the .zip file.
- In the Terminal, activate the virtual environment:

   $ source /vol/lab/intro2ml/venv/bin/activate
- In the virtual environent, install the required packages:
  
  $ python3 -c "import numpy as np; import torch; print(np); print(torch)"

- Navigate to the source code file directory in the terminal.
- Run the command: python3 decisionTree.py - to run the Python script.
- Once run, code will display the confusion matrices, the evaluation metrics for both datasets and plot the decision tree for the clean dataset.

Repository contains:
- Clean & Noisy datasets in wifi_db
- Script decisionTree.py to build decision tree & run evaluation

