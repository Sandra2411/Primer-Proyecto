# Primer Proyecto Individual
## Descripción del Proyecto

Este proyecto se centra en el proceso ETL (Extracción, transformación y carga) de datos relacionados con juegos de Steam. Además, incluye un análisis exploratorio de datos (EDA) y la implementación de cinco funciones, así como un modelo de recomendación de juegos. 

1. PlayTimeGenre: Esta función devuelve el año con la mayor cantidad de horas jugadas para un género de juego específico. Proporciona información valiosa sobre la popularidad a lo largo del tiempo de un género en particular.

2. UserForGenre: La función devuelve el usuario que ha acumulado la mayor cantidad de horas jugadas para un género de juego dado. Además, proporciona una lista que muestra la acumulación de horas jugadas por año para ese género, lo que ayuda a identificar tendencias a lo largo del tiempo.

3. UsersRecommend: Esta función proporciona una lista de los tres juegos más recomendados por los usuarios para un año específico. Considera las revisiones marcadas como "recomendadas" y los comentarios positivos o neutrales.

4. UsersNotRecommend: A diferencia de la función anterior, esta devuelve los tres juegos menos recomendados por los usuarios para un año dado. Se basa en revisiones marcadas como "no recomendadas" y comentarios negativos.

5. sentiment_analysis: Esta función analiza el sentimiento de las revisiones de usuarios y devuelve una lista que muestra la cantidad de registros de reseñas categorizados según su análisis de sentimiento para un año específico.

- Modelo de Recomendación: Se  implementa un modelo de recomendación que se basa en la similitud entre juegos. Al ingresar el ID de un juego, el modelo devuelve una lista de 5 juegos recomendados que son similares al juego ingresado. Este modelo se basa en las características y preferencias de los usuarios.

Este proyecto con juegos  brinda a los usuarios la capacidad de obtener recomendaciones personalizadas. Los resultados de estas funciones y del modelo de recomendación pueden ser utilizados para tomar decisiones informadas en la industria de los videojuegos y comprender mejor las preferencias de los usuarios a lo largo del tiempo.

## Extracción, Transformación y Carga (ETL) de Datos 

En este proyecto, se lleva a cabo un proceso ETL para tres archivos relacionados con datos de juegos de Steam. Cada uno de estos archivos pasa por un proceso de extracción, limpieza y transformación antes de ser almacenado en formato Parquet. 

### Archivo Steam Games Data

###### Extracción y Descompresión:
Se inicia extrayendo y descomprimiendo el archivo "steam_games.json.gz". Los datos son leídos línea por línea y almacenados en un DataFrame.

###### Limpieza de Datos:
Se eliminan filas con datos faltantes y se restablecen los índices del DataFrame para asegurar su integridad.

###### Transformación de Fechas:
La columna "release_date" se convierte en un tipo de dato datetime para facilitar el análisis de fechas. La columna "price" se convierte a tipo numérico.

###### Almacenamiento en Formato Parquet:
Finalmente, se crea un archivo Parquet llamado "games.parquet" que contiene los datos limpios y transformados.

###  Archivo - User Reviews Data

Extracción y Descompresión: 
El archivo "user_reviews.json.gz" se descomprime y se almacenan los datos en un DataFrame.

###### Limpieza de Caracteres No Válidos:
Se eliminan caracteres no válidos en los datos para asegurar la integridad del DataFrame.

###### Desanidamiento de Diccionarios:
La columna "reviews" contiene diccionarios anidados que se desanidan para facilitar el análisis.

###### Limpieza de Datos Faltantes:
Se eliminan las filas con datos faltantes y se restablecen los índices del DataFrame.

###### Transformación de Fechas:
Se realiza una transformación de la columna "posted" para obtener el mes, día y año de publicación.

###### Análisis de Sentimiento:
Se aplica un análisis de sentimiento a las reseñas de usuarios y se agrega la columna "sentiment_analysis" al DataFrame.

###### Almacenamiento en Formato Parquet: 
Los datos limpios se guardan en un archivo Parquet llamado "reviews.parquet".

### Archivo - Users Items Data

###### Extracción y Descompresión: 
El archivo "users_items.json.gz" se descomprime y los datos se almacenan en un DataFrame.

###### Limpieza de Datos No Válidos:
Se eliminan caracteres no válidos en los datos para garantizar la integridad del DataFrame.

###### Desanidamiento de Listas:
Las listas anidadas se desanidan para facilitar el análisis.

###### Selección de Columnas Relevantes:
Se seleccionan las columnas pertinentes para el análisis.

###### Limpieza de Datos Faltantes:
Se eliminan las filas con datos faltantes y se restablecen los índices del DataFrame.

###### Transformación de Datos:
La columna "item_id" se convierte a tipo numérico.

###### Almacenamiento en Formato Parquet:
Los datos limpios se almacenan en un archivo Parquet llamado
"items.parquet".

# Eda PlayTimeGenre
Se carga un archivo .parquet llamado 'games.parquet' en un DataFrame llamado 'df_games' y se muestra información básica sobre el DataFrame.

Se seleccionan campos específicos ('genres', 'release_date', 'id') de 'df_games' y se almacenan en un nuevo DataFrame llamado 'df_games_subset'. Además, se seleccionan campos de otros DataFrames llamados 'df_reviews' y 'df_items'.

Se combinan los DataFrames seleccionados ('df_games', 'df_reviews', 'df_items') en un nuevo DataFrame llamado 'df_funcion_play' utilizando la función join.

La columna 'genres' en 'df_funcion_play'  se desglosan las listas de géneros en filas individuales. Luego, se cuenta la cantidad de juegos en cada género y se almacena en 'genero_counts'.

Se obtiene una lista de años únicos en la columna 'release_date', se cuentan y se muestra el número total de años únicos.

Se obtiene una lista de años únicos en la columna 'release_date', se cuenta la cantidad de juegos por año y se muestra en un gráfico de barras horizontal.

Se crea un DataFrame 'lanzamientos_por_año' que muestra la cantidad de lanzamientos de juegos por año. Luego, se visualiza esta información en un gráfico de barras.

Se agrupan los datos por año de lanzamiento y se suma el tiempo de juego en minutos. Se crea un DataFrame 'minutos_por_año' que muestra los minutos totales de juego por año y se visualiza en un gráfico de barras.

Se expanden las listas de géneros en nuevas filas, se agrega una columna con el año de lanzamiento y se crea una tabla dinámica ('tabla_minutos_genero') que muestra los minutos jugados por género y año de lanzamiento.

Se define una función llamada 'PlayTimeGenre' que toma un género como entrada, filtra las filas correspondientes a ese género y encuentra el año con más horas jugadas para ese género. Luego, se imprime el resultado para el género "Casual".

Se crea un archivo .parquet llamado 'data_play.parquet' que contiene datos relevantes para la función 'PlayTimeGenre'.

Se devuelve el DataFrame 'df_play', que contiene información sobre géneros, fechas de lanzamiento y tiempo de juego.


##  EDA UserForGenre EDA UserForGenre 
Creo dataframe de mis archivos para las columnas que necesito para esta funcion

Se combinan tres DataFrames: 'df_games', 'df_reviews_', y 'df_items, utilizando la función merge en dos etapas para obtener un nuevo DataFrame llamado 'df_funcion_genre'. Los DataFrames se unen por sus índices.

Se eliminan las filas que contienen datos faltantes (NaN) del DataFrame 'df_funcion_genre' utilizando la función dropna() y se muestra la información actualizada del DataFrame.

Se reorganizan los índices del DataFrame después de eliminar los datos faltantes utilizando df_funcion_genre.reset_index(drop=True, inplace=True).

Se define una función llamada 'UserForGenre' que toma un DataFrame y un género como entrada. Esta función filtra las filas que contienen el género especificado, encuentra al usuario que jugó más minutos para ese género y agrupa los minutos jugados por año.

Se llama a la función 'UserForGenre' con el DataFrame 'df_funcion_genre' y el género. Se almacena el usuario que jugó más horas para ese género y se crea una lista de acumulación de horas jugadas por año.

Se imprime el usuario con más horas jugadas para el género y se muestra la acumulación de horas jugadas por año.

Se crea un archivo .parquet llamado 'data_genre.parquet' que contiene datos relevantes para la función', incluyendo géneros, fechas de lanzamiento, tiempo de juego, ID de artículo y ID de usuario.

##  EDA UsersRecommend

Se combinan las columnas de diferentes DataFrames ('df_games', 'df_reviews', 'df_items') en un nuevo DataFrame llamado 'funcion_recommend' utilizando la función pd.concat(). Las columnas incluyen información sobre títulos de juegos, recomendaciones, fechas de publicación y análisis de sentimiento, así como identificadores de usuarios.

Se muestra información básica sobre el DataFrame 'funcion_recommend' utilizando funcion_recommend.info().

Se eliminan las filas que contienen datos faltantes (NaN) del DataFrame 'funcion_recommend' utilizando funcion_recommend.dropna() y se muestra la información actualizada del DataFrame.

Se reorganizan los índices del DataFrame después de eliminar los datos faltantes utilizando funcion_recommend.reset_index(drop=True, inplace=True).

Se cambia el tipo de dato de la columna 'recommend' a booleano utilizando funcion_recommend['recommend'] = funcion_recommend['recommend'].astype(bool).

Se filtran los datos en dos grupos: recomendaciones verdaderas ('True') y recomendaciones falsas ('False'). Luego, se cuentan las cantidades de usuarios en cada grupo y se visualizan en un gráfico de barras.

Se definen las categorías y etiquetas para el análisis de sentimiento, se calcula el conteo de cada categoría en la columna 'sentiment_analysis' y se renombran las columnas. Luego, se muestra la distribución de sentimiento en un gráfico de barras.

Se define la función 'UsersRecommend' que toma un año y un número n como entrada. Esta función filtra los juegos recomendados con comentarios positivos o neutrales para el año especificado, agrupa por título, cuenta las recomendaciones y selecciona los juegos más recomendados (por defecto, los 3 primeros). El resultado se almacena en una lista.

Se llama a la función 'UsersRecommend' para el año 2013 y se muestra la lista de los juegos más recomendados para ese año.

Se crea un archivo .parquet llamado 'data_recommend.parquet' que contiene datos relevantes para la función 'UsersRecommend', incluyendo información sobre recomendaciones, títulos, análisis de sentimiento y fechas de publicación.

## EDA. UsersNotRecommend
Se crea un nuevo DataFrame llamado 'funcion_norecommend' mediante la combinación de columnas de diferentes DataFrames ('df_games', 'df_reviews', 'df_items') utilizando la función pd.concat(). Las columnas incluyen información sobre títulos de juegos, recomendaciones, fechas de publicación y análisis de sentimiento, así como identificadores de usuarios. Se muestra información básica sobre el DataFrame 'funcion_norecommend' utilizando funcion_norecommend.info().

Se eliminan las filas que contienen datos faltantes (NaN) del DataFrame 'funcion_norecommend' utilizando funcion_norecommend.dropna() y se muestra la información actualizada del DataFrame.

Se reorganizan los índices del DataFrame después de eliminar los datos faltantes utilizando funcion_norecommend.reset_index(drop=True, inplace=True).

Se cambia el tipo de dato de la columna 'recommend' a booleano utilizando funcion_norecommend['recommend'] = funcion_norecommend['recommend'].astype(bool).

Se definen las categorías y etiquetas para el análisis de sentimiento y se calcula el conteo de cada categoría en la columna 'sentiment_analysis', que es un proceso similar al realizado previamente en la función 'UsersRecommend'.

Se seleccionan campos requeridos para la función 'sentiment_analysis' y se concatenan en un nuevo DataFrame llamado 'funcion_sentiment'. Luego, se muestra información sobre este DataFrame.

Se define la función 'UsersNotRecommend' que toma un año y un número n como entrada. Esta función filtra los juegos no recomendados con comentarios negativos (sentimiento igual a 0) para el año especificado, agrupa por título, cuenta las no recomendaciones y selecciona los juegos menos recomendados (por defecto, los 3 primeros). El resultado se almacena en una lista.

Se llama a la función 'UsersNotRecommend' para el año 2013 y se muestra la lista de los juegos menos recomendados para ese año.

## EDA. sentiment_analysis

Se eliminan las filas que contienen datos faltantes (NaN) del DataFrame 'funcion_sentiment' utilizando funcion_sentiment.dropna() y se muestra la información actualizada del DataFrame.

Se reorganizan los índices del DataFrame después de eliminar los datos faltantes utilizando funcion_sentiment.reset_index(drop=True, inplace=True).

Se extrae el año de lanzamiento de la fecha de 'release_date' y se calcula el total de sentimientos por año. Luego, se muestra un gráfico de barras que representa el total de sentimientos por año de lanzamiento.

Se crea un DataFrame llamado 'sentiment_by_year_df' que contiene información sobre el año de lanzamiento y el total de sentimientos por año. Este DataFrame se muestra.

Se calcula el número de sentimientos por año y categoría (negativos, neutros, positivos) y se muestra un gráfico de barras apiladas que representa el número de sentimientos por año de lanzamiento y categoría.

Se renombra el DataFrame resultante como 'sentiments_by_year' para que las columnas tengan nombres más descriptivos, y se muestra el DataFrame.

Se define una función llamada 'sentiment_analysis' que toma un año y un DataFrame como entrada. La función filtra el DataFrame por el año especificado, calcula el número de registros para cada categoría de análisis de sentimiento y crea un diccionario con los resultados. Luego, se muestra la cantidad de registros para cada categoría.

Se llama a la función 'sentiment_analysis' para el año 2012 y el DataFrame 'funcion_sentiment', mostrando la cantidad de registros para cada categoría de análisis de sentimiento.

Se crea un archivo .parquet llamado 'data_sentiment.parquet' que contiene datos relevantes para el análisis de sentimiento, incluyendo información sobre el análisis de sentimiento, fechas de publicación, ID de artículo y ID de usuario.

Se muestra una vista previa de las primeras filas del DataFrame 'df_sentiment' para verificar los datos guardados en el archivo .parquet.

EDA Modelo de Recomendación:

Se seleccionan los campos necesarios para el modelo de machine learning, que incluyen 'genres' y 'title' de juegos de un DataFrame llamado 'df_games', así como 'item_id' de un DataFrame 'df_items'. Estos campos se concatenan en un nuevo DataFrame llamado 'df_juego', y se muestra información sobre él.

Se eliminan las filas que contienen datos faltantes (NaN) del DataFrame 'df_juego' utilizando df_juego.dropna() y se muestra la información actualizada del DataFrame.

Se reorganizan los índices del DataFrame después de eliminar los datos faltantes utilizando df_juego.reset_index(drop=True, inplace=True).

Se crea una matriz TF-IDF de los géneros de los juegos utilizando la biblioteca scikit-learn. Luego, se calcula la similitud de coseno entre los juegos basándose en sus géneros.

Se define una función llamada 'recommend_games_by_genre' que toma un 'item_id' y el DataFrame como entrada. Esta función calcula la puntuación de similitud de coseno para todos los juegos con el juego dado, los ordena según su puntuación de similitud y devuelve los juegos más similares.

Se llama a la función 'recommend_games_by_genre' con un 'item_id' específico y se muestra una lista de juegos recomendados basados en géneros similares.

Se filtran y seleccionan los campos relevantes para el modelo en el DataFrame 'df_juego', que incluyen 'genres', 'item_id' y 'title'.

Se crea un archivo .parquet llamado 'data_juego.parquet' que contiene datos relevantes para el modelo, incluyendo información sobre géneros, identificadores de juegos y títulos.
