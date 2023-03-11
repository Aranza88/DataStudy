"""
    Universidad Nacional Autónoma de México
    Facultad de Ingeniería
    Inteligencia Artificial Gpo. 5
    Proyecto final: DataStudy
    Autora: Núñez Luna, Aranza Abril
    No. de cuenta: 317079867
    Correo: aranzaabril88@gmail.com
"""

import streamlit as st
import pandas as pd  # Para la manipulación y análisis de datos
import numpy as np  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt  # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist  # Para el cálculo de distancias
from scipy.spatial import distance
import seaborn as sb
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def metricasDistancia():
    st.markdown("# Métricas de distancia")

    st.markdown("## ¿Qué son?")
    st.markdown('<p align="justify", style="font-size: 20px;">Las métricas de distancia consisten en un valor objetivo que resume la diferencia entre dos elementos. Estas por lo general son numéricas y establecen que tan cercanos son dos objetos. Por ejemplo, si en un plano hay dos puntos la distancia será el espacio entre ellos.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">Las métricas de distancia no son algoritmos de Inteligencia Artificial como tal, pero son muy utilizados en este tipo de algoritmos para toda clase de cosas. En esta implementación, puedes observar cuatro distintas distancias y las matrices de distancias resultantes. En las matrices de distancia, cada valor corresponde a la relación entre la variable en la columna y la de la fila, así que es bastante sencilla de leer.</p>', unsafe_allow_html=True)

    st.markdown("## Características de los datos")
    st.markdown('<p align="justify", style="font-size: 20px;">Los datos para este algoritmo deben estar organizados en un documento csv con los nombres de cada variable como encabezado. Si el documento no tiene encabezado, puede que el primer valor se tome como nombre de variable, así que ¡ten cuidado!</p>', unsafe_allow_html=True)

    docMD=None
    docMD=st.file_uploader("Cargue su archivo en formato .csv", type=["csv"], key="MD")

    if docMD is not None:
        #Se cargan los datos
        st.markdown("### Datos insertados")
        DataMD = pd.read_csv(docMD)
        st.write(DataMD)

        #Se imprime el mapa de calor de los datos originales
        fig, ax = plt.subplots(figsize=(15, 8))
        HeatMp = np.triu(DataMD.corr())
        g1 = sb.heatmap(DataMD.corr(), cmap="RdBu_r", annot=True, mask=HeatMp, ax=ax)
        st.pyplot(fig)

        #Se pide al usuario realizar la selección de variables
        varSelect=st.multiselect("Seleccione variables a eliminar (Si no desea eliminar variables, solo continue.)", DataMD.columns.values)
        vars_object=list(DataMD.columns[DataMD.dtypes == 'object'])

        #Se eliminan las variables seleccionadas
        DataDrop=DataMD.drop(columns=vars_object)
        NewData=DataDrop.drop(columns=varSelect)

        #Se imprimen los datos modificados
        st.markdown("### Datos modificados")
        st.write(NewData)
        fig2,ax2=plt.subplots(figsize=(15, 8))
        HeatMp2=np.triu(NewData.corr())
        g2=sb.heatmap(NewData.corr(), cmap="RdBu_r", annot=True, mask=HeatMp2, ax=ax2)
        st.pyplot(fig2)
            
        #Se estandarizan los datos
        estandarizar = StandardScaler()  # Se instancia el objeto StandardScaler o MinMaxScaler
        MEstandarizada = estandarizar.fit_transform(NewData)  # Se calculan la media y desviación y se escalan los datos
        st.write("Los datos se han estandarizado.")

        #Se selecciona la métrica de distancia a utilizar
        st.markdown("# Selección de métrica")
        st.markdown('<p align="justify", style="font-size: 20px;">Ahora que los datos están listos, selecciona la métrica que quieras utilizar. Cada una de ellas se explicará cuando la selecciones.</p>', unsafe_allow_html=True)
        metrica=st.selectbox("Seleccione una métrica de distancia", ["-","Euclidiana", "Chebyshev", "Manhattan", "Minkowski"])
        
        if metrica=="Euclidiana":
            st.markdown("## Métrica de distancia euclidiana")
            st.markdown('<p align="justify", style="font-size: 20px;">Esta métrica de distancia es de las más utilizadas y se basa en el Teorema de Pitágoras. Es así que la distancia vendría a ser la hipotenusa de un triángulo rectángulo formado a partir de la posición de los puntos. La fórmula es la siguiente:</p>', unsafe_allow_html=True)
            col11, col12, col13 = st.columns(3)
            col12.image("img\MDE.png", width=200)
            st.markdown('<p align="justify", style="font-size: 20px;">De los datos proporcionados, esta es la matriz de distancias resultante con esta métrica de distancia:</p>', unsafe_allow_html=True)
            DstEuclidiana=cdist(MEstandarizada, MEstandarizada, metric='euclidean')
            MEuclidiana = pd.DataFrame(DstEuclidiana)
            st.dataframe(MEuclidiana.round(4))
        elif metrica=="Chebyshev":
            st.markdown("## Métrica de distancia Chebyshev")
            st.markdown('<p align="justify", style="font-size: 20px;">La distancia de Chebyshev, también conocida como métrica máxima, consiste en el valor máximo absoluto de las diferencias entre las coordenadas de un par de elementos. Esto quiere decir que calcula la distancia directamente de la posición de cada valor. La fórmula es la siguiente:</p>', unsafe_allow_html=True)
            col21, col22, col23 = st.columns(3)
            col22.image("img\MDC.png", width=200)
            st.markdown('<p align="justify", style="font-size: 20px;">De los datos proporcionados, esta es la matriz de distancias resultante con esta métrica de distancia:</p>', unsafe_allow_html=True)
            DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
            MChebyshev = pd.DataFrame(DstChebyshev)
            st.dataframe(MChebyshev.round(4))
        elif metrica=="Manhattan":
            st.markdown("## Métrica de distancia Manhattan")
            st.markdown('<p align="justify", style="font-size: 20px;">La distancia de Manhattan suele utilizarse para distancias geoespaciales, pues calcula la distancia de dos puntos en una ruta parecida a la de las calles de Manhattan, es decir, una cuadrícula. La fórmula que se utiliza es la siguiente:</p>', unsafe_allow_html=True)
            col31, col32, col33 = st.columns(3)
            col32.image("img\MDMan.png", width=200)
            st.markdown('<p align="justify", style="font-size: 20px;">De los datos proporcionados, esta es la matriz de distancias resultante con esta métrica de distancia:</p>', unsafe_allow_html=True)
            DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
            MManhattan = pd.DataFrame(DstManhattan)
            st.dataframe(MManhattan.round(4))
        elif metrica=="Minkowski":
            st.markdown("## Métrica de distancia Minkowski")
            st.markdown('<p align="justify", style="font-size: 20px;">La distancia de Monkowski es una distancia peculiar, pues busca ser generalizada en todo sentido. Esta distancia se calcula en un espacio n-dimensional y de forma que cualquiera de las otras distancias en DataStudy se pueden utilizar en este algoritmo. Es la más compleja, pero aquí te presentamos la fórmula:</p>', unsafe_allow_html=True)
            col41, col42, col43, col44 = st.columns(4)
            col42.image("img\MDMin.png", use_column_width='never', width=400)
            st.markdown('<p align="justify", style="font-size: 20px;">De los datos proporcionados, esta es la matriz de distancias resultante con esta métrica de distancia:</p>', unsafe_allow_html=True)
            DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
            MMinkowski = pd.DataFrame(DstMinkowski)
            st.dataframe(MMinkowski)
        else:
            st.write("**Seleccione una métrica de distancia para continuar.**")