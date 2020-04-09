"""
This program is used for data preprocessing
@author: tyeh3
"""
# Press Command + Enter to execute
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

pd.options.mode.chained_assignment = None

# Import the dataset (assuming the working directory is CSC522Project)
# credits_data = credits_data[credits_data['id'].isin(set(movies_meta_data['id'].unique()))]
# credits_data.to_csv('data/casts.csv', index=False)
crews_data = pd.read_csv('data/crews.csv')
casts_data = pd.read_csv('data/casts.csv')
keywords_data = pd.read_csv('data/keywords.csv')
movies_meta_data = pd.read_csv('data/movies_metadata.csv')

# Select the movie data that is useful
movies_meta_data = movies_meta_data.dropna(subset=['budget'])
movies_meta_data = movies_meta_data.dropna(subset=['revenue'])
movies_meta_data = movies_meta_data.replace(np.nan, '', regex=True)
movies_meta_data = movies_meta_data.loc[:, ['budget', 'genres', 'id', 'overview', 'popularity', 'release_date', 'revenue', 'runtime', 'tagline', 'vote_average', 'vote_count']]
movies_meta_data['budget'] = pd.to_numeric(movies_meta_data['budget'], errors='coerce')
movies_meta_data['vote_average'] = pd.to_numeric(movies_meta_data['vote_average'], errors='coerce')
movies_meta_data['vote_count'] = pd.to_numeric(movies_meta_data['vote_count'], errors='coerce')
movies_meta_data['revenue'] = pd.to_numeric(movies_meta_data['revenue'], errors='coerce')
movies_meta_data['runtime'] = pd.to_numeric(movies_meta_data['runtime'], errors='coerce')
movies_meta_data['revenue'] = pd.to_numeric(movies_meta_data['revenue'], errors='coerce')
movies_meta_data['id'] = pd.to_numeric(movies_meta_data['id'], downcast='unsigned')
movies_meta_data['popularity'] = pd.to_numeric(movies_meta_data['popularity'])
movies_meta_data['release_date']= pd.to_datetime(movies_meta_data['release_date'])

# Select the movies with both revenue and budget greater than 0
movies_meta_data = movies_meta_data[((movies_meta_data.loc[:,'budget'] > 0) & (movies_meta_data.loc[:,'revenue'] > 0))]
# Remove the movie with runtime equals 0
movies_meta_data = movies_meta_data[movies_meta_data['runtime'] > 0]
# Calculate return on investment attributes
movies_meta_data['return_on_investment'] = ((movies_meta_data['revenue'] - movies_meta_data['budget']) / movies_meta_data['budget'])
# Remove movie with return on investment greater than 1000 based on historical data
movies_meta_data = movies_meta_data[movies_meta_data['return_on_investment'] < 1000]

# Merge with the keywords and credits
movies_meta_data = movies_meta_data.merge(keywords_data, left_on='id', right_on='id')
movies_meta_data = movies_meta_data.merge(casts_data, left_on='id', right_on='id')
movies_meta_data = movies_meta_data.merge(crews_data, left_on='id', right_on='id')
del(keywords_data)
del(casts_data)
del(crews_data)

# Drop the unnecessary column
movies_meta_data = movies_meta_data.drop(columns=['id','vote_count'])

# Generate the new feature for genres, keywords, and cast
genres_name_count_map = dict()
genres_name_total_pop_map = dict()
genres_name_total_vote_map = dict()
for genres in [eval(movie) for movie in movies_meta_data['genres']]:
    for g in genres:
        genres_name_count_map.update({g['name']: genres_name_count_map.get(g['name'], 0) + 1})
        
movies_meta_data['genres'] = [[g['name'] for g in genres] for genres in [eval(movie) for movie in movies_meta_data['genres']]]
del(genres)

keywords_name_count_map = dict()
keywords_name_total_pop_map = dict()
keywords_name_total_vote_map = dict()
for kws in [eval(keyword) for keyword in movies_meta_data['keywords']]:
    for k in kws:
        keywords_name_count_map.update({k['name']: keywords_name_count_map.get(k['name'], 0) + 1})
movies_meta_data['keywords'] = [[k['name'] for k in keyword] for keyword in [eval(keywords) for keywords in movies_meta_data['keywords']]]
del(kws)

casts_name_count_map = dict()
casts_name_total_pop_map = dict()
casts_name_total_vote_map = dict()
for cast in [eval(casts) for casts in movies_meta_data['cast']]:
    for c in cast:
        casts_name_count_map.update({c['name']: casts_name_count_map.get(c['name'], 0) + 1})
movies_meta_data['cast'] = [[c['name'] for c in cast] for cast in [eval(casts) for casts in movies_meta_data['cast']]]
del(cast)

directors_name_count_map = dict()
directors_name_total_pop_map = dict()
directors_name_total_vote_map = dict()
for crew in [eval(crews) for crews in movies_meta_data['crew']]:
    for c in crew:
        if c['job'] == 'Director':
            directors_name_count_map.update({c['name']: directors_name_count_map.get(c['name'], 0) + 1})
movies_meta_data['directors'] = [[c['name'] for c in crew if c['job'] == 'Director'] for crew in [eval(crews) for crews in movies_meta_data['crew']]]
movies_meta_data = movies_meta_data.drop(columns=['crew'])
del(crew)

for i in movies_meta_data.index: 
     for g in movies_meta_data['genres'][i]:
         genres_name_total_pop_map.update({g: genres_name_total_pop_map.get(g, 0) + movies_meta_data['popularity'][i]})
         genres_name_total_vote_map.update({g: genres_name_total_vote_map.get(g, 0) + movies_meta_data['vote_average'][i]})
     for k in movies_meta_data['keywords'][i]:
         keywords_name_total_pop_map.update({k: keywords_name_total_pop_map.get(k, 0) + movies_meta_data['popularity'][i]})
         keywords_name_total_vote_map.update({k: keywords_name_total_vote_map.get(k, 0) + movies_meta_data['vote_average'][i]})
     for c in movies_meta_data['cast'][i]:
         casts_name_total_pop_map.update({c: casts_name_total_pop_map.get(c, 0) + movies_meta_data['popularity'][i]})
         casts_name_total_vote_map.update({c: casts_name_total_vote_map.get(c, 0) + movies_meta_data['vote_average'][i]})
     for d in movies_meta_data['directors'][i]:
         directors_name_total_pop_map.update({d: directors_name_total_pop_map.get(d, 0) + movies_meta_data['popularity'][i]})
         directors_name_total_vote_map.update({d: directors_name_total_vote_map.get(d, 0) + movies_meta_data['vote_average'][i]})

genres_name_avg_pop_map = dict()
genres_name_avg_vote_map = dict()
keywords_name_avg_pop_map = dict()
keywords_name_avg_vote_map = dict()
casts_name_avg_pop_map = dict()
casts_name_avg_vote_map = dict()
directors_name_avg_pop_map = dict()
directors_name_avg_vote_map = dict()
for g in genres_name_count_map:
    genres_name_avg_pop_map.setdefault(g, genres_name_total_pop_map.get(g) / genres_name_count_map.get(g))
    genres_name_avg_vote_map.setdefault(g, genres_name_total_vote_map.get(g) / genres_name_count_map.get(g))
for k in keywords_name_count_map:
    keywords_name_avg_pop_map.setdefault(k, keywords_name_total_pop_map.get(k) / keywords_name_count_map.get(k))
    keywords_name_avg_vote_map.setdefault(k, keywords_name_total_vote_map.get(k) / keywords_name_count_map.get(k))
for c in casts_name_count_map:
    casts_name_avg_pop_map.setdefault(c, casts_name_total_pop_map.get(c) / casts_name_count_map.get(c))
    casts_name_avg_vote_map.setdefault(c, casts_name_total_vote_map.get(c) / casts_name_count_map.get(c))
for d in directors_name_count_map:
    directors_name_avg_pop_map.setdefault(d, directors_name_total_pop_map.get(d) / directors_name_count_map.get(d))
    directors_name_avg_vote_map.setdefault(d, directors_name_total_vote_map.get(d) / directors_name_count_map.get(d))
del(i, g, k, c, d)
del(genres_name_count_map, genres_name_total_pop_map, genres_name_total_vote_map)
del(keywords_name_count_map, keywords_name_total_pop_map, keywords_name_total_vote_map)
del(casts_name_count_map, casts_name_total_pop_map, casts_name_total_vote_map)
del(directors_name_count_map, directors_name_total_pop_map, directors_name_total_vote_map)

movies_meta_data['genres_popularity_score'] = 0.0
movies_meta_data['genres_vote_score'] = 0.0
movies_meta_data['keywords_popularity_score'] = 0.0
movies_meta_data['keywords_vote_score'] = 0.0
movies_meta_data['casts_popularity_score'] = 0.0
movies_meta_data['casts_vote_score'] = 0.0
movies_meta_data['directors_popularity_score'] = 0.0
movies_meta_data['directors_vote_score'] = 0.0
for i in movies_meta_data.index: 
    # genres
    g_pop_sum = 0
    g_vote_sum = 0
    g_count = 0
    for g in movies_meta_data['genres'][i]:
        g_pop_sum = g_pop_sum + genres_name_avg_pop_map.get(g)
        g_vote_sum = g_vote_sum + genres_name_avg_vote_map.get(g)
        g_count = g_count + 1
    if g_count == 0:
        movies_meta_data['genres_popularity_score'][i] = 0
        movies_meta_data['genres_vote_score'][i] = 0
    else:
        movies_meta_data['genres_popularity_score'][i] = g_pop_sum / g_count
        movies_meta_data['genres_vote_score'][i] = g_vote_sum / g_count
    
    # keywords
    k_pop_sum = 0
    k_vote_sum = 0
    k_count = 0
    for k in movies_meta_data['keywords'][i]:
        k_pop_sum = k_pop_sum + keywords_name_avg_pop_map.get(k)
        k_vote_sum = k_vote_sum + keywords_name_avg_vote_map.get(k)
        k_count = k_count + 1 
    if k_count == 0:
        movies_meta_data['keywords_popularity_score'][i] = 0
        movies_meta_data['keywords_vote_score'][i] = 0
    else:
        movies_meta_data['keywords_popularity_score'][i] = k_pop_sum / k_count
        movies_meta_data['keywords_vote_score'][i] = k_vote_sum / k_count
    
    # casts
    c_pop_sum = 0
    c_vote_sum = 0
    c_count = 0
    for c in movies_meta_data['cast'][i]:
        c_pop_sum = c_pop_sum + casts_name_avg_pop_map.get(c)
        c_vote_sum = c_vote_sum + casts_name_avg_vote_map.get(c)
        c_count = c_count + 1 
    if c_count == 0:
        movies_meta_data['casts_popularity_score'][i] = 0
        movies_meta_data['casts_vote_score'][i] = 0
    else:
        movies_meta_data['casts_popularity_score'][i] = c_pop_sum / c_count
        movies_meta_data['casts_vote_score'][i] = c_vote_sum / c_count
    
    # directors
    d_pop_sum = 0
    d_vote_sum = 0
    d_count = 0
    for d in movies_meta_data['directors'][i]:
        d_pop_sum = d_pop_sum + directors_name_avg_pop_map.get(d)
        d_vote_sum = d_vote_sum + directors_name_avg_vote_map.get(d)
        d_count = d_count + 1 
    if d_count == 0:
        movies_meta_data['directors_popularity_score'][i] = 0
        movies_meta_data['directors_vote_score'][i] = 0
    else:
        movies_meta_data['directors_popularity_score'][i] = d_pop_sum / d_count
        movies_meta_data['directors_vote_score'][i] = d_vote_sum / d_count

del(i)
del(c, c_count, c_pop_sum, c_vote_sum)
del(g, g_count, g_pop_sum, g_vote_sum)
del(k, k_count, k_pop_sum, k_vote_sum)
del(d, d_count, d_pop_sum, d_vote_sum)
del(genres_name_avg_pop_map, genres_name_avg_vote_map)
del(keywords_name_avg_pop_map, keywords_name_avg_vote_map)
del(casts_name_avg_pop_map, casts_name_avg_vote_map)
del(directors_name_avg_pop_map, directors_name_avg_vote_map)

# Remove the outliers using z-score
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.genres_popularity_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.genres_vote_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.keywords_popularity_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.keywords_vote_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.casts_popularity_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.casts_vote_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.directors_popularity_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.directors_vote_score)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.budget)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.revenue)) <= 3)]
movies_meta_data = movies_meta_data[(np.abs(stats.zscore(movies_meta_data.return_on_investment)) <= 3)]

movies_meta_data = movies_meta_data.drop(columns=['budget','revenue'])
movies_meta_data = movies_meta_data.drop(columns=['genres','keywords','cast','directors','overview','tagline','popularity','vote_average'])
movies_meta_data.info()

movies_meta_data['release_year'] = 0
movies_meta_data['release_month'] = 0
for i in movies_meta_data.index: 
    movies_meta_data['release_year'][i] = movies_meta_data['release_date'][i].year
    movies_meta_data['release_month'][i] = movies_meta_data['release_date'][i].month

desc_all = movies_meta_data.describe(include='all')
for d in desc_all:
    print()
    print("{}:\nmin={:.3f}\nmax={:.3f}\nmedian={:.3f}\nmean={:.3f}\nstd.dev={:.3f}".format(d, desc_all[d]['min'], desc_all[d]['max'], desc_all[d]['50%'], desc_all[d]['mean'], desc_all[d]['std']))
del(d, i, release_date)