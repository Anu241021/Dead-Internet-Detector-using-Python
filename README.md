# Dead Internet Detector

An NLP-powered web application that classifies text as **Human-written** or **AI/Bot-generated** using traditional machine learning techniques. The project leverages TF-IDF vectorization, engineered text features, and a Logistic Regression classifier, and is deployed through Streamlit for real-time predictions.

## Features

- Classifies text as Human-written or AI/Bot-generated
- End-to-end machine learning pipeline for text preprocessing, feature extraction, training, and inference
- TF-IDF vectorization with additional engineered text features
- Interactive web interface built with Streamlit
- Fast real-time predictions using a trained Logistic Regression model

## Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy

## How It Works

1. Accepts user input through the Streamlit interface.
2. Cleans and preprocesses the input text.
3. Converts text into numerical features using TF-IDF vectorization.
4. Extracts additional handcrafted text features.
5. Uses a trained Logistic Regression model to classify the text.
6. Displays the prediction instantly through the web interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Dead-Internet-Detector.git
cd Dead-Internet-Detector
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Future Improvements

- Improve model performance using larger and more diverse datasets
- Experiment with transformer-based models such as BERT
- Display prediction confidence scores
- Support multilingual text classification
- Integrate explainable AI techniques for prediction interpretation

## Contributing

Contributions are welcome. Feel free to fork the repository, open an issue, or submit a pull request.

## License

This project is licensed under the MIT License.
