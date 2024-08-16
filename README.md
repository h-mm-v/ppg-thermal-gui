# pathsup_rice2024
Contains codes developed during the PATHS-UP Summer 2024 REU program at Rice University

<h3 align="center">Graphical User Interface for Extracting Vitals from CMS50d+ Pulse Oximeters and Thermal Cameras</h3>

  <p align="center">
    A multimodal graphical user interface to record synchronous data from a pulse oximeter and thermal camera.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<p>Vital signs, notably respiration rate, heart rate, and heart rate variability, are important measures for assessing health status. Photoplethysmography (PPG) is a low-cost and noninvasive technique that can be processed to obtain such vital measurements. For PPG signals, we developed a peak detection algorithm that is robust to corruption. These extrema points are essential to obtain numerical estimates for the pulse - namely the heart rate and heart rate variability. Building on this, we processed these extrema to infer the respiratory rate from the PPG signal. Specifically, we explore and analyze the frequency profile of HRV, peak amplitude, and signal intensity to estimate the respiratory signal [1].</p>
<p>Parallely, we explore thermal imaging to develop a non-contact-based method to monitor RR. This is a crucial step forwards as contact-based methods can cause discomfort to patients and alter breathing rates. Current non-contact methods have room for improvement given their sensitivity to extraneous movements. We estimate RR from thermal videos by monitoring temperature changes in the region around the nose caused by inhaling cold air and exhaling warm air. This area is extracted using machine learning methods to detect 54 facial landmarks [2], and hence the region of interest (ROI). Respiration being quasi-periodic, can be identified through a frequency-based analysis of the ROI. Specifically, we look at ways to subdivide the ROI and quantify its quality through measures of periodicity and signal strength.</p>
<p>Combining both modules, we develop an end-to-end system to acquire and analyze thermal and PPG signals in order to estimate the participant’s breathing rate. We placed the camera to be facing the participant at angle from underneath the face to better capture the temperature variations near the nose. We tested several recordings with the subject in different positions and breathing at different rates. The estimate from the thermal camera, the rate calculated from our PPG-based algorithm, and the estimation from directly counting breaths in the recorded video were all consistent with each other.</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* scipy
  ```sh
  py pip install scipy
  ```
* customtkinter
  ```sh
  py pip install customtkinter
  ```
* CTkMessagebox
  ```sh
  py pip install CTkMessagebox
  ```
* numpy
  ```sh
  py pip install numpy
  ```
* tkinter
  ```sh
  py pip install tkinter
  ```

* matplotlib
  ```sh
  py pip install matplotlib
  ```

### Installation

1. Download ppg_analysis.py and ppg_thermal_GUI.py

2. Run ppg_thermal_GUI.py

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


_For an example, please refer to the [Demo](https://github.com/user-attachments/assets/90b642dd-54cf-4f7a-a3f5-820381a50196)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Butterworth bandpass filter
- [x] Peak Enhancement algorithm
- [x] Fiducial Point Identification
- [x] Biomarker Calculation
- [x] Vital Sign Estimation 

<p align="right">(<a href="#readme-top">back to top</a>)</p>
