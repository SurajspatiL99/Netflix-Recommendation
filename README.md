# Netflix Recommender System

![image](https://github.com/SurajspatiL99/Netflix-Recommendation-System/assets/101862962/5f1d90f7-2473-476f-a3ea-a63c6bbfbc08)

This project is a Netflix recommender system built using collaborative filtering techniques. The goal of the system is to provide personalized movie recommendations to users based on their preferences and similarities with other users. The system utilizes a bag-of-words approach to generate similarity scores between two movies and then recommends movies based on these scores. Additionally, a web app is built using Streamlit to provide a user-friendly interface, and a dashboard is created on Power BI to visualize the recommendation results.

## Dataset

The dataset used for this project is obtained from [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv). It consists of anonymized movie ratings from a large number of users. The dataset is preprocessed to extract relevant information such as movie titles and ratings, which are used to build the recommender system.

## Prerequisites

To run the code and reproduce the recommender system, you need the following prerequisites:

- Python 3.x
- Jupyter Notebook or any Python IDE of your choice
- Streamlit
- Power BI Desktop

The following Python libraries are required as well:

- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib

These libraries can be installed using pip with the following command:

```
pip install pandas numpy scikit-learn streamlit matplotlib
```

## Usage

1. Clone this repository to your local machine or download and extract the ZIP file.

```
git clone https://github.com/SurajspatiL99/Netflix-Recommendation-System.git
```

2. Navigate to the project directory.

```
cd netflix-recommender-system
```

3. Launch Jupyter Notebook or your preferred Python IDE.

```
jupyter notebook
```

4. Open the `netflix-recommender-system.ipynb` file.

5. Execute the code cells in sequential order to run the recommender system.

6. To run the web app, navigate to the project directory in your terminal and execute the following command:

```
streamlit run app.py
```

7. To view the Power BI dashboard, open Power BI Desktop and import the provided dataset. Then, open the `Dashboard.pbix` file.

## Project Structure

The project is structured as follows:

- `netflix-recommender-system.ipynb`: Jupyter Notebook containing the code for building the recommender system using collaborative filtering.

- `app.py`: Streamlit web app code for the user interface of the recommender system.

- `dataset.csv`: Preprocessed dataset used for building the recommender system.

- `dashboard.pbix`: Power BI dashboard file for visualizing the recommendation results.

![Screenshot 2023-03-09 203543](https://user-images.githubusercontent.com/101862962/224200813-82de4f03-864f-4605-a0f6-e94b1b5a3fce.png)


## Acknowledgements

- The Netflix dataset used in this project was obtained from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv)

- This project is for educational purposes and inspired by the open-source community.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Contact

For questions, suggestions, or further information about the project, please contact LinkedIn.

Enjoy personalized movie recommendations from Netflix!
