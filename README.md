<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]

<h3 align="center">Graphical User Interface for Extracting Vitals from CMS50d+ Pulse Oximeters and Thermal Cameras</h3>

  <p align="center">
    A multimodal graphical user interface to record synchronous data from a pulse oximeter and thermal camera.
    <br />
    <a href="https://github.com/hmmv/ppg-thermal-gui"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/hmmv/ppg-thermal-gui">View Demo</a>
    ·
    <a href="https://github.com/hmmv/ppg-thermal-gui/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/hmmv/ppg-thermal-gui/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
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
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

  Vital signs, notably respiration rate, heart rate, and heart rate variability, are important measures for assessing health status. Photoplethysmography (PPG) is a low-cost and noninvasive technique that can be processed to obtain such vital measurements. For PPG signals, we developed a peak detection algorithm that is robust to corruption. These extrema points are essential to obtain numerical estimates for the pulse - namely the heart rate and heart rate variability. Building on this, we processed these extrema to infer the respiratory rate from the PPG signal. Specifically, we explore and analyze the frequency profile of HRV, peak amplitude, and signal intensity to estimate the respiratory signal [1]. 
  Parallely, we explore thermal imaging to develop a non-contact-based method to monitor RR. This is a crucial step forwards as contact-based methods can cause discomfort to patients and alter breathing rates. Current non-contact methods have room for improvement given their sensitivity to extraneous movements. We estimate RR from thermal videos by monitoring temperature changes in the region around the nose caused by inhaling cold air and exhaling warm air. This area is extracted using machine learning methods to detect 54 facial landmarks [2], and hence the region of interest (ROI). Respiration being quasi-periodic, can be identified through a frequency-based analysis of the ROI. Specifically, we look at ways to subdivide the ROI and quantify its quality through measures of periodicity and signal strength. 
  Combining both modules, we develop an end-to-end system to acquire and analyze thermal and PPG signals in order to estimate the participant’s breathing rate. We placed the camera to be facing the participant at angle from underneath the face to better capture the temperature variations near the nose. We tested several recordings with the subject in different positions and breathing at different rates. The estimate from the thermal camera, the rate calculated from our PPG-based algorithm, and the estimation from directly counting breaths in the recorded video were all consistent with each other.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/hmmv/ppg-thermal-gui.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```
3. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/hmmv/ppg-thermal-gui/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Henry Vo - henry.minhm.vo@gmail.com

Project Link: [https://github.com/hmmv/ppg-thermal-gui](https://github.com/hmmv/ppg-thermal-gui)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/hmmv/ppg-thermal-gui.svg?style=for-the-badge
[contributors-url]: https://github.com/hmmv/ppg-thermal-gui/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hmmv/ppg-thermal-gui.svg?style=for-the-badge
[forks-url]: https://github.com/hmmv/ppg-thermal-gui/network/members
[stars-shield]: https://img.shields.io/github/stars/hmmv/ppg-thermal-gui.svg?style=for-the-badge
[stars-url]: https://github.com/hmmv/ppg-thermal-gui/stargazers
[issues-shield]: https://img.shields.io/github/issues/hmmv/ppg-thermal-gui.svg?style=for-the-badge
[issues-url]: https://github.com/hmmv/ppg-thermal-gui/issues
[license-shield]: https://img.shields.io/github/license/hmmv/ppg-thermal-gui.svg?style=for-the-badge
[license-url]: https://github.com/hmmv/ppg-thermal-gui/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/henry-minh-man-vo
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
