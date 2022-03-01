# sheet-midi-sync
Code and data repository for ISMIR 2019 paper: MIDI–SHEET MUSIC ALIGNMENT USING BOOTLEG SCORE SYNTHESIS

## Notebooks
1. PrepScoreData: This notebook preprocesses the sheet music data. It convert multiple-page pdf sheet music into image of musical strips with empty border removed.
2. SheetMidiSync: This is the main notebook of this project. It converts MIDI into bootleg score and convert sheet music into notehead blobs before aligning them with DTW. The outputs are saved into a newly created hyp_align folder to be evaluated by notebook 3.
3. Evaluate_Bootleg: This notebook evaluates the performance of the bootleg system using annotation in annot_data folder in the form of error tolerance with time interval and horizontal pixel distance as error metrics. It also contains the code for global linear baseline system.
4. Baseline: This notebook contains implementation of MIDI-beat-matching and audio-sync baseline systems which runs on OMR-generated midi in the synth_midi folder. It compares the performances of these baseline systems to bootleg system evaluation from notebook 3.

## Data folders
- midi: contains midi files for experiment
- score_data: contains sheet music (in pdf format) for experiment
- synth_midi: contains midi files generated from sheet music in score_data folder with 2 commercial optical music recognition software, PhotoScore and SharpEye, for running MIDI-beat-matching and Audio Sync baseline systems.
- annot_data: annotation for all data above in csv format, used in notebook 3 and 4.

## Running the notebooks

### Language
Python 3

### Dependencies
- Numpy 1.16.2
- ImageMagick 
- Keras 2.2.4 
- tensorflow-gpu 1.8.0
- opencv-python 3.4.5.20
- Pillow 5.4.1
- scipy 1.2.1
- scikit-learn 0.20.3
- librosa 0.6.3 
- mido 1.2.9
- pretty_midi 0.2.8

## Additional notes
The sheet_id folder contains the code needed to run notehead detection with deep-convolutional neural network. Download the neural network model weight from a separate source (https://www.dropbox.com/s/m5gc90vodpy8wce/dwd-re-finetune-20000.h5?dl=0) and put it in sheet_id folder to run the notebooks.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
