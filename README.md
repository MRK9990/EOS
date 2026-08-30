# EOS

### Dental Image Color Normalization & Prediction System

EOS is an independent research project focused on the analysis and
normalization of tooth color in dental photographs under varying
lighting and environmental conditions.

The project explores how machine learning can be used to estimate
and compensate for image acquisition conditions in order to make
dental photographs more comparable.

---

## About the Project

Dental photographs are strongly affected by acquisition conditions.

Differences in:

- illumination;
- white balance;
- exposure;
- contrast;
- camera settings;
- background color;
- environmental conditions;

can significantly change the apparent color of teeth.

EOS is designed to study and reduce these effects through a
machine-learning-based image normalization pipeline.

The current version uses a supervised machine learning approach
based on Random Forest and MultiOutputRegressor.

The model predicts a set of image correction parameters, including:

- white balance;
- red gain;
- blue gain;
- brightness;
- contrast;
- exposure.

These parameters are then used to produce a normalized image.

---

## Current Architecture

The project is currently organized into several main components:

```text
EOS/
│
├── apps/       # User-facing applications
├── configs/    # Configuration files
├── docs/       # Project documentation
├── scripts/    # Dataset, training and evaluation scripts
├── src/        # Core EOS source code
├── tests/      # Tests
├── data/       # Local/private dataset (not included)
└── models/     # Trained models (not included)


# Machine Learning Pipeline

The current research pipeline consists of:

Dental photograph
        │
        ▼
Image preprocessing
        │
        ▼
Feature extraction
        │
        ▼
EOS machine learning model
        │
        ▼
Predicted correction parameters
        │
        ▼
Image normalization
        │
        ▼
Normalized dental photograph

The current model is intentionally relatively lightweight and
interpretable compared with deep neural-network approaches.

This makes it suitable for experimentation, rapid iteration and
evaluation during the early stages of the project.


## Dataset

The EOS dataset is currently maintained locally and is not included
in this repository.

The dataset contains dental photographs organized according to
image acquisition conditions and associated metadata.

The decision not to publish the dataset is intentional and is related
to data management, privacy and research considerations.

Therefore, the repository contains the code required to construct and
process the dataset, but not the underlying image collection.


## Model

Trained EOS model files are not included in this repository.

The repository contains the training and evaluation code required to
reproduce the machine-learning pipeline when an appropriate dataset
is available.

Model files generated locally are stored in:

models/

and are excluded from version control.


## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd EOS
```

Create a virtual environment:

**Windows**

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
py -m pip install -r requirements.txt
```


## Running EOS

**Dataset construction**

```bash
py scripts/build_dataset.py
```

**Model training**

```bash
py scripts/train_model.py
```

**Model evaluation**

```bash
py scripts/evaluate_model.py
```

**Predictor application**

```bash
py apps/predictor_gui.py
```

The predictor requires a locally available trained EOS model.


## Research Status

EOS is an ongoing independent research project.

The current implementation should be considered a research prototype
rather than a finalized clinical system.

The model is being developed incrementally, with future work focused
on improving robustness, generalization and performance under
different image acquisition conditions.


## Future Development

Planned research directions include:

- expansion of the training dataset;
- improved feature engineering;
- stronger validation procedures;
- robustness to different cameras and lighting conditions;
- comparison with alternative machine learning models;
- investigation of computer vision and deep learning approaches;
- improved color-space representations;
- uncertainty estimation;
- more rigorous experimental evaluation.


## Author

EOS is developed independently by a single researcher.

I am a PhD student working on the project independently, including
software development, dataset preparation, machine learning,
experimentation and system design.

The project is developed incrementally as an independent research
initiative.


## Disclaimer

EOS is a research and experimental software project.

It is not intended to provide medical or dental diagnoses,
treatment recommendations, or replace professional clinical
assessment.

Color normalization results may depend on image quality, lighting,
camera characteristics and other acquisition conditions.


## License

The source code of EOS is released under the MIT License.

The EOS dataset, trained model files and other non-public research
assets are not included in this repository and are not covered by
the public code distribution.

See LICENSE for the full license text.