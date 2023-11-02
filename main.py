import pandas as pd
import fastapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



app = fastapi.FastAPI()


# def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.

df_funcion_play = pd.read_parquet ("data_play.parquet")

@app.get("/play_time_genre/{genero}", response_model=dict)

def play_time_genre(genero: str):
    
    df_expansion = df_funcion_play.explode('genres')

    df_filas = df_expansion[df_expansion['genres'] == genero]

    if df_filas.empty:
        return {
            "message": f"No se encontraron datos para el género {genero}",
            "max_year": None
        }

    max_year = df_filas.groupby('release_date')['playtime_forever'].sum().idxmax().year

    response_data = {
        f"Año con más horas jugadas para el género {genero}": max_year
    }

    return response_data



# def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

df_funcion_genre = pd.read_parquet("data_genre.parquet")

@app.get("/user_for_genre/{genero}", response_model=dict)
def user_for_genre(genero: str):
    
    df_filtered = df_funcion_genre[df_funcion_genre['genres'].apply(lambda x: genero in x)]

    if df_filtered.empty:
        return {
            "message": f"No se encontraron datos para el género {genero}",
            "acumulacion_horas_por_anio": []
        }

    usuario_max_horas = df_filtered[df_filtered['playtime_forever'] == df_filtered['playtime_forever'].max()]['user_id'].values[0]

    acumulacion_horas_por_anio = df_filtered.groupby(df_filtered['release_date'].dt.year)['playtime_forever'].sum()

    response_data = {
        "Usuario con más horas jugadas": usuario_max_horas,
        "acumulacion_horas_por_anio": [{"Año": año, "Horas": horas} for año, horas in acumulacion_horas_por_anio.items()]
    }

    return response_data







# def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)

funcion_recommend = pd.read_parquet("data_recommend.parquet")

@app.get("/users_recommend/{year}")
def users_recommend(year: int, n: int = 3):
    
    filtered_df = funcion_recommend[(funcion_recommend['recommend'] == True) & (funcion_recommend['sentiment_analysis'] >= 0) & (funcion_recommend['posted_date'].dt.year == year)]

    if filtered_df.empty:
        return "No se encontraron datos para el año y condiciones especificadas", []

    recommended_by_title = filtered_df.groupby('title')['recommend'].sum().reset_index()
    recommended_by_title = recommended_by_title.sort_values(by='recommend', ascending=False)

    top_games = recommended_by_title.head(n)

    result = [{"Puesto": i + 1, "Juego": game} for i, game in enumerate(top_games['title'])]

    return result


# def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)

funcion_norecommend = pd.read_parquet ("data_norecommend.parquet")

@app.get("/users_not_recommend/{year}")

def users_not_recommend(year: int, n: int = 3):
    
    filtered_df = funcion_norecommend[(funcion_norecommend['recommend'] == False) & (funcion_norecommend['sentiment_analysis'] == 0) & (funcion_norecommend['posted_date'].dt.year == year)]

    not_recommended_by_title = filtered_df.groupby('title')['recommend'].count().reset_index()
    not_recommended_by_title = not_recommended_by_title.sort_values(by='recommend', ascending=False)

    top_games = not_recommended_by_title.head(n)

    result = [{"Puesto": i + 1, "Juego": game} for i, game in enumerate(top_games['title'])]

    return result


# def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

dataframe = pd.read_parquet ("data_sentiment.parquet")

@app.get("/sentiment_analysis/{year}")
def get_sentiment_analysis(year: int):
    
    filtrado_por_año = dataframe[dataframe['release_date'].dt.year == year]

    resultados = filtrado_por_año['sentiment_analysis'].value_counts().to_dict()

    resultado = {
        'Negative': resultados.get(0, 0),
        'Neutral': resultados.get(1, 0),
        'Positive': resultados.get(2, 0)
    }

    return resultado



# def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

df_juegos = pd.read_parquet ("data_juegos.parquet")

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_juegos['genres'].apply(lambda x: ' '.join(x)))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.get("/recomendacion_juego/{item_id}")
async def recomendacion_juego(item_id: int):
    
    idx = df_juegos[df_juegos['item_id'] == item_id].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    game_indices = [i[0] for i in sim_scores]
    recommended_games = df_juegos['title'].iloc[game_indices].tolist()

    return {"recommended_games": recommended_games}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
