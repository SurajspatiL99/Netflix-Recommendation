# Netflix Recommendation System


```python
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
#reading in the Dataset
df = pd.read_csv('netflix_titles.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Kirsten Johnson</td>
      <td>NaN</td>
      <td>United States</td>
      <td>September 25, 2021</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>NaN</td>
      <td>Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban...</td>
      <td>South Africa</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, TV Dramas, TV Mysteries</td>
      <td>After crossing paths at a party, a Cape Town t...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>Julien Leclercq</td>
      <td>Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi...</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Crime TV Shows, International TV Shows, TV Act...</td>
      <td>To protect his family from a powerful drug lor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Docuseries, Reality TV</td>
      <td>Feuds, flirtations and toilet talk go down amo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>NaN</td>
      <td>Mayur More, Jitendra Kumar, Ranjan Raj, Alam K...</td>
      <td>India</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, Romantic TV Shows, TV ...</td>
      <td>In a city of coaching centers known to train I...</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis


```python
# Count of Null values in every column
df.isnull().sum()
```




    show_id            0
    type               0
    title              0
    director        2634
    cast             825
    country          831
    date_added        10
    release_year       0
    rating             4
    duration           3
    listed_in          0
    description        0
    dtype: int64




```python
# Replacing all Null values with 'Na' 
df = df.fillna('Na')
```


```python
# Helper function to extract genres and cast from the colum "listed_in" and "cast" respectively
# As a movie or show can have multiple genres and cast members we will make seperate dataframe for both of them

def convert_to_series(dataframe, y):
    genres = []
    for row in dataframe[y]:
        row_modified = row.split(",")
        for genre in row_modified:
            genres.append(genre)
            
    genres_df = pd.DataFrame(genres)
    return genres_df

# Contains all the Genres from all the movies and shows
genres= convert_to_series(df, "listed_in")

# Contains all the Cast Members names from all the movies and shows
cast = convert_to_series(df, "cast")
```

## Ploting the Top 10 Genres from this dataset 


```python
genres[0].value_counts().head(11)[1:].plot.bar()
```




    <AxesSubplot:>




    
![png](output_9_1.png)
    



```python
cast = cast[cast[0] !='Na']
cast.reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ama Qamata</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Khosi Ngema</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gail Mabalane</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thabang Molaba</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dillon Windvogel</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>64121</th>
      <td>Manish Chaudhary</td>
    </tr>
    <tr>
      <th>64122</th>
      <td>Meghna Malik</td>
    </tr>
    <tr>
      <th>64123</th>
      <td>Malkeet Rauni</td>
    </tr>
    <tr>
      <th>64124</th>
      <td>Anita Shabdish</td>
    </tr>
    <tr>
      <th>64125</th>
      <td>Chittaranjan Tripathy</td>
    </tr>
  </tbody>
</table>
<p>64126 rows × 1 columns</p>
</div>



## Ploting the Top 10 Cast Members with most movies or shows


```python
cast.value_counts().head(20).plot.bar()
```




    <AxesSubplot:xlabel='0'>




    
![png](output_12_1.png)
    


## Ploting the Top 10 Directors with most movies or shows


```python
df['director'].value_counts().head(11)[1:].plot.bar()
```




    <AxesSubplot:>




    
![png](output_14_1.png)
    


## Ploting the Movies or Shows with particular Age Restriction


```python
df['rating'].value_counts().head(10).plot.bar()
```




    <AxesSubplot:>




    
![png](output_16_1.png)
    


## Count of movies and shows in the data set


```python
df['type'].value_counts().plot.pie()
```




    <AxesSubplot:ylabel='type'>




    
![png](output_18_1.png)
    


## Top Countries where most Netflix Movies or shows are produced 


```python
df['country'].value_counts().head(6).plot.pie()
```




    <AxesSubplot:ylabel='country'>




    
![png](output_20_1.png)
    


# Recommendation System 


```python

```


```python
df.columns
```




    Index(['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
           'release_year', 'rating', 'duration', 'listed_in', 'description'],
          dtype='object')




```python
df = df[['show_id', 'type', 'title', 'director', 'cast',
       'release_year', 'listed_in', 'description']]
```


```python
def Convert(string):
    l1 = []
    for i in string:
        li = list(i.split(","))
        l1.append(li)
    return l1
```


```python
df['director'] = (Convert(df['director']))
df['cast'] = (Convert(df['cast']))
df['listed_in'] = (Convert(df['listed_in']))
```


```python
df['description'] = df['description'].apply(lambda x:x.split())
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>release_year</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>[Kirsten Johnson]</td>
      <td>[Na]</td>
      <td>2020</td>
      <td>[Documentaries]</td>
      <td>[As, her, father, nears, the, end, of, his, li...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>[Na]</td>
      <td>[Ama Qamata,  Khosi Ngema,  Gail Mabalane,  Th...</td>
      <td>2021</td>
      <td>[International TV Shows,  TV Dramas,  TV Myste...</td>
      <td>[After, crossing, paths, at, a, party,, a, Cap...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>[Julien Leclercq]</td>
      <td>[Sami Bouajila,  Tracy Gotoas,  Samuel Jouy,  ...</td>
      <td>2021</td>
      <td>[Crime TV Shows,  International TV Shows,  TV ...</td>
      <td>[To, protect, his, family, from, a, powerful, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>[Na]</td>
      <td>[Na]</td>
      <td>2021</td>
      <td>[Docuseries,  Reality TV]</td>
      <td>[Feuds,, flirtations, and, toilet, talk, go, d...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>[Na]</td>
      <td>[Mayur More,  Jitendra Kumar,  Ranjan Raj,  Al...</td>
      <td>2021</td>
      <td>[International TV Shows,  Romantic TV Shows,  ...</td>
      <td>[In, a, city, of, coaching, centers, known, to...</td>
    </tr>
  </tbody>
</table>
</div>




```python
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
```


```python
df['director'] = df['director'].apply(collapse)
df['cast'] = df['cast'].apply(collapse)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>release_year</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>[KirstenJohnson]</td>
      <td>[Na]</td>
      <td>2020</td>
      <td>[Documentaries]</td>
      <td>[As, her, father, nears, the, end, of, his, li...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>[Na]</td>
      <td>[AmaQamata, KhosiNgema, GailMabalane, ThabangM...</td>
      <td>2021</td>
      <td>[International TV Shows,  TV Dramas,  TV Myste...</td>
      <td>[After, crossing, paths, at, a, party,, a, Cap...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>[JulienLeclercq]</td>
      <td>[SamiBouajila, TracyGotoas, SamuelJouy, Nabiha...</td>
      <td>2021</td>
      <td>[Crime TV Shows,  International TV Shows,  TV ...</td>
      <td>[To, protect, his, family, from, a, powerful, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>[Na]</td>
      <td>[Na]</td>
      <td>2021</td>
      <td>[Docuseries,  Reality TV]</td>
      <td>[Feuds,, flirtations, and, toilet, talk, go, d...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>[Na]</td>
      <td>[MayurMore, JitendraKumar, RanjanRaj, AlamKhan...</td>
      <td>2021</td>
      <td>[International TV Shows,  Romantic TV Shows,  ...</td>
      <td>[In, a, city, of, coaching, centers, known, to...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['tags'] = df['director'] + df['cast'] + df['listed_in'] + df['description']
```


```python
new_df = df[['show_id', 'type', 'title', 'release_year', 'tags']]
```


```python
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>release_year</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>2020</td>
      <td>[KirstenJohnson, Na, Documentaries, As, her, f...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>2021</td>
      <td>[Na, AmaQamata, KhosiNgema, GailMabalane, Thab...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>2021</td>
      <td>[JulienLeclercq, SamiBouajila, TracyGotoas, Sa...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>2021</td>
      <td>[Na, Na, Docuseries,  Reality TV, Feuds,, flir...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>2021</td>
      <td>[Na, MayurMore, JitendraKumar, RanjanRaj, Alam...</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
```

    C:\Users\suraj\AppData\Local\Temp\ipykernel_4656\1824047427.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    


```python
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

    C:\Users\suraj\AppData\Local\Temp\ipykernel_4656\1380776331.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    


```python
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>release_year</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>2020</td>
      <td>kirstenjohnson na documentaries as her father ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>2021</td>
      <td>na amaqamata khosingema gailmabalane thabangmo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>2021</td>
      <td>julienleclercq samibouajila tracygotoas samuel...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>2021</td>
      <td>na na docuseries  reality tv feuds, flirtation...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>2021</td>
      <td>na mayurmore jitendrakumar ranjanraj alamkhan ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
```


```python
vector = cv.fit_transform(new_df['tags']).toarray()
```


```python

```


```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
```


```python
def stem (text):
    y =[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
```


```python
new_df['tags'] = new_df['tags'].apply(stem)
```

    C:\Users\suraj\AppData\Local\Temp\ipykernel_4656\3213734980.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      new_df['tags'] = new_df['tags'].apply(stem)
    


```python
vector = cv.fit_transform(new_df['tags']).toarray()

```


```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
similarity = cosine_similarity(vector)
```


```python
similarity.shape
```




    (8807, 8807)




```python
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[0:6]:
        print(new_df.iloc[i[0]]['title'], new_df.iloc[i[0]]['release_year'])

```


```python
recommend("Friends")
```

    Friends 2003
    Why Are You Like This 2021
    Dad's Army 1977
    Workin' Moms 2021
    Little Things 2019
    La Rosa de Guadalupe 2010
    


```python

```


```python

```


```python

```
