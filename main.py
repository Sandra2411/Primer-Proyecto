import pandas as pd
import fastapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



app = fastapi.FastAPI()



# Supongamos que tienes un DataFrame df_funcion_play que contiene tus datos


df_funcion_play = pd.read_parquet ("data_play.parquet")

@app.get("/play_time_genre/{genero}", response_model=dict)

def play_time_genre(genero: str):
    # Asegúrate de que df_funcion_play tenga una columna llamada 'genres'
    df_expansion = df_funcion_play.explode('genres')

    # Filtra las filas que contienen el género especificado
    df_filas = df_expansion[df_expansion['genres'] == genero]

    if df_filas.empty:
        return {
            "message": f"No se encontraron datos para el género {genero}",
            "max_year": None
        }

    # Encuentra el año con más horas jugadas para el género
    max_year = df_filas.groupby('release_date')['playtime_forever'].sum().idxmax().year

    response_data = {
        f"Año de lanzamiento con más horas jugadas para el género {genero}": max_year
    }

    return response_data




# Reemplaza 'data_genre.parquet' con la ruta correcta a tu archivo de datos
df_funcion_genre = pd.read_parquet("data_genre.parquet")

@app.get("/user_for_genre/{genero}", response_model=dict)
def user_for_genre(genero: str):
    # Realiza el filtro de género en tu DataFrame df_funcion_genre (reemplaza esto por tu propio DataFrame).
    df_filtered = df_funcion_genre[df_funcion_genre['genres'].apply(lambda x: genero in x)]

    if df_filtered.empty:
        return {
            "message": f"No se encontraron datos para el género {genero}",
            "acumulacion_horas_por_anio": []
        }

    # Encuentra el usuario con más horas jugadas.
    usuario_max_horas = df_filtered[df_filtered['playtime_forever'] == df_filtered['playtime_forever'].max()]['user_id'].values[0]

    # Calcula la acumulación de horas por año.
    acumulacion_horas_por_anio = df_filtered.groupby(df_filtered['release_date'].dt.year)['playtime_forever'].sum()

    # Formatea la respuesta en un diccionario.
    response_data = {
        "Usuario con más horas jugadas": usuario_max_horas,
        "acumulacion_horas_por_anio": [{"Año": año, "Horas": horas} for año, horas in acumulacion_horas_por_anio.items()]
    }

    return response_data







# Supongamos que tienes un DataFrame llamado funcion_recommend que contiene tus datos


# Carga el DataFrame desde el archivo "data_recommend.parquet"
funcion_recommend = pd.read_parquet("data_recommend.parquet")

@app.get("/users_recommend/{year}")
def users_recommend(year: int, n: int = 3):
    # Filtra los juegos recomendados y con comentarios positivos/neutrales para el año especificado
    filtered_df = funcion_recommend[(funcion_recommend['recommend'] == True) & (funcion_recommend['sentiment_analysis'] >= 0) & (funcion_recommend['posted_date'].dt.year == year)]

    if filtered_df.empty:
        return "No se encontraron datos para el año y condiciones especificadas", []

    # Agrupa por título, cuenta las recomendaciones y ordena en orden descendente
    recommended_by_title = filtered_df.groupby('title')['recommend'].sum().reset_index()
    recommended_by_title = recommended_by_title.sort_values(by='recommend', ascending=False)

    # Selecciona los juegos más recomendados (los 3 primeros por defecto)
    top_games = recommended_by_title.head(n)

    # Crea una lista de los juegos más recomendados en formato vertical
    result = [{"Puesto": i + 1, "Juego": game} for i, game in enumerate(top_games['title'])]

    return result




funcion_norecommend = pd.read_parquet ("data_norecommend.parquet")

# Supongamos que tienes un DataFrame llamado funcion_norecommend que contiene tus datos

@app.get("/users_not_recommend/{year}")

def users_not_recommend(year: int, n: int = 3):
    # Filtrar los juegos no recomendados y con comentarios negativos para el año especificado
    filtered_df = funcion_norecommend[(funcion_norecommend['recommend'] == False) & (funcion_norecommend['sentiment_analysis'] == 0) & (funcion_norecommend['posted_date'].dt.year == year)]

    # Agrupar por título, contar las no recomendaciones y ordenar en orden descendente
    not_recommended_by_title = filtered_df.groupby('title')['recommend'].count().reset_index()
    not_recommended_by_title = not_recommended_by_title.sort_values(by='recommend', ascending=False)

    # Seleccionar los juegos menos recomendados (los 3 primeros por defecto)
    top_games = not_recommended_by_title.head(n)

    # Crear una lista de los juegos menos recomendados en formato vertical
    result = [{"Puesto": i + 1, "Juego": game} for i, game in enumerate(top_games['title'])]

    return result


dataframe = pd.read_parquet ("data_sentiment.parquet")

@app.get("/sentiment_analysis/{year}")
def get_sentiment_analysis(year: int):
    # Filtrar el DataFrame por el año de lanzamiento
    filtrado_por_año = dataframe[dataframe['release_date'].dt.year == year]

    # Calcular el número de registros para cada categoría de análisis de sentimiento
    resultados = filtrado_por_año['sentiment_analysis'].value_counts().to_dict()

    # Crear un diccionario con los resultados
    resultado = {
        'Negative': resultados.get(0, 0),
        'Neutral': resultados.get(1, 0),
        'Positive': resultados.get(2, 0)
    }

    return resultado




# Supongamos que tienes tus datos cargados en el DataFrame df_juegos
df_juegos = pd.read_parquet ("data_juegos.parquet")
# Paso 1: Crear una matriz TF-IDF de los géneros de los juegos
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_juegos['genres'].apply(lambda x: ' '.join(x)))

# Paso 2: Calcular la similitud de coseno entre los juegos
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.get("/recomendacion_juego/{item_id}")
async def recomendacion_juego(item_id: int):
    # Encontrar el índice del juego en función de su item_id
    idx = df_juegos[df_juegos['item_id'] == item_id].index[0]

    # Calcular la puntuación de similitud de coseno para todos los juegos con el juego dado
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar los juegos según su puntuación de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los juegos más similares (excluyendo el juego dado)
    sim_scores = sim_scores[1:6]

    # Obtener los títulos de los juegos recomendados
    game_indices = [i[0] for i in sim_scores]
    recommended_games = df_juegos['title'].iloc[game_indices].tolist()

    return {"recommended_games": recommended_games}

# Ejemplo de uso: Accede a la recomendación de juegos mediante http://localhost:8000/recomendacion_juego/{item_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
