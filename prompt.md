I’m building an audio event classifier for an ESP32-S3 wearable (TFLite Micro). Please set up this repo and write the code. Read everything first, then build it in the layout below.

Environment: Python, TensorFlow/Keras. Training machine is a laptop with an RTX 3050 6GB. It must run on GPU if available and fall back to CPU. Keep memory use low (small batches, cached features, no loading everything at once). Create a requirements.txt and note that GPU TensorFlow needs WSL2/Linux.

Data: ESC-50 is in data/ESC-50/ (audio/ and meta/esc50.csv). Don’t touch the dataset files.

Design goal: the training code must be independent of where audio came from and how many microphones were used. Build it as three stages that communicate through a standardized dataset folder. Small warning, you may have to search the codebase for some of the specific files.

1. scripts/prepare_esc50.py

Read esc50.csv. Map classes from config.yaml (default: siren, car_horn, alarm = clock_alarm, knock = door_wood_knock; everything else = background).
Convert to 16 kHz mono, cut into 1 s windows (16,000 samples), skip near-silent windows with an energy check (do this for target classes only, not background), and write to data/processed/<class>/<name>.wav.
Write data/processed/manifest.csv with columns: path, class, source (“esc50” or “own”), source_file_id, split. For ESC-50 use folds 1-3 = train, 4 = val, 5 = test.
Subsample background to about 4x the size of the average target class.

2. scripts/convert_recordings.py (for my own recordings)

Input: files in my_recordings_raw/<class>/ with any sample rate and any channel count (1, 2, 4, etc.).
CLI options: --channel N (use one specific channel), --mix (average all channels), default is to use channel 0. Log a warning if the file has multiple channels.
Resample to 16 kHz, convert to mono, normalize safely, cut into 1 s windows, and write to data/processed/<class>/ and append to manifest.csv.
Split by source file, never by window. All windows from one recording go to the same split (about 70/15/15 across files), so there’s no leakage.
Support any class name in the folder, so I can add classes like doorbell later without changing code.

3. train.py (reads only manifest.csv and config.yaml)

Classes come from the manifest/config, not hardcoded.
Features: TensorFlow audio_microfrontend op, 40 channels, 20 ms step, 30 ms window, 16 kHz, output shape about 49x40x1. Cache computed features to disk (npy) so repeated runs are fast.
Augmentation on the training split only: background noise mixing from the background class at random SNR (0-20 dB), random gain, random time shift within the window. Keep it memory-safe, either by generating augmented copies once or in a tf.data pipeline.
Model: DS-CNN (one conv layer, four depthwise-separable blocks with 64 filters, global average pooling, softmax), under 80k parameters. Print the parameter count.
Adam, cross-entropy with class weights, early stopping on val loss, up to 50 epochs. Seed everything.
After training, on the test split print per-class precision and recall, a confusion matrix, and overall accuracy. Save the confusion matrix as a PNG. Also report the false-positive rate on the background test windows.
This is meant to train on cpu only, and to be as lightweight as possible in training as well in performance while not compromising efficiency.

4. export.py

Save the Keras model, then export a float TFLite and an int8-quantized TFLite using a representative dataset from the training features.
Evaluate both on the test set and print the accuracy of each, with a warning if int8 is more than 3 points lower.
Write the int8 model to model.h as a C array, and save the class label list to labels.txt.
Print the int8 input scale and zero-point, since the firmware needs them.

General: put all settings (classes, sample rate, window length, feature params, split ratios, seeds) in config.yaml. Add a README.md with the exact commands to run each step in order. Add clear error messages and small unit tests for the audio slicing and the manifest split logic. Don’t add dependencies beyond what’s needed.