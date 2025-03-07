# MOS Analyzer (Experimental Project)

This is an **experimental Android application** that integrates **DNSMOS** to calculate the **Mean Opinion Score (MOS)** for an audio file. The app allows users to **select an audio file, compute MOS using ONNX Runtime, and display the result in a user-friendly UI** built with **Jetpack Compose**.

## Features
- **Audio File Selection:** Choose an audio file from the device storage.
- **MOS Calculation:** Computes MOS using a deep learning model.
- **Chaquopy Integration:** Runs Python scripts for audio feature extraction.
- **ONNX Runtime:** Performs inference on the pre-trained DNSMOS model.
- **Modern UI:** Uses Jetpack Compose for a responsive and dynamic interface.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/MOS-Analyzer.git
    cd MOS-Analyzer
2. Open the project in Android Studio.
3. Ensure the required dependencies are installed in build.gradle.
4. Place the model_v8.onnx file inside app/src/main/assets/.
5. Run the application on an Android device or emulator.

## Technologies Used
- **Kotlin & Jetpack Compose** – For Android development
- **Chaquopy (Python in Android)** – For feature extraction
- **ONNX Runtime** – For running DNSMOS on mobile
- **Material 3 Design** – For a sleek UI

## Notes
- This is an **experimental project** and may not always provide accurate results.
- The MOS calculation is based on the **pre-trained DNSMOS model**, and its accuracy depends on the model's training data.