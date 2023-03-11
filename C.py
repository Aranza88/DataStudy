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
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sb             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
st.set_option('deprecation.showPyplotGlobalUse', False)

def cluestering():
    st.markdown("# Clustering")

    st.markdown("## ¿Qué es?")
    st.markdown('<p align="justify", style="font-size: 20px;">Para comprender el clustering, es necesario comprender que es un clúster. Un clúster es un conjunto de elementos similares o clasificados de una misma forma. Por ejemplo, si hay un conjunto de pelotas de colores, las pelotas azules serían un clúster, las rojas otro, etc.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">El clustering consiste en una serie de algoritmos que tiene como objetivo separar a los datos en clústeres, clasificándolos según sus características. Existen dos tipos de clustering: el clustering jerárquico y el clustering particional. Sin embargo, ambos siguen los siguientes pasos:</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Utilizar un método para medir la similitud de los elementos. En este caso se utilizan métricas de distancia que puedes seleccionar. Para saber más sobre métricas de distancia puedes ir a la página correspondiente a través de la barra lateral.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Utilizar un método para agrupar elementos. Este método depende del tipo de clustering.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Utilizar un método para decidir la cantidad adecuada de grupos. Esta cantidad se puede seleccionar o utilizar un método para decidir.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Interpretación de grupos. Esto dependerá en gran medida de los datos que utilices, pero recuerda que todos los elementos de un clúster tienen algo en común, ya sea que estén dentro de un rango de valores o estén clasificados de alguna forma.</p>', unsafe_allow_html=True)

    st.markdown("## Características del documento")
    st.markdown('<p align="justify", style="font-size: 20px;">Los datos para este algoritmo deben estar organizados en un documento csv con los nombres de cada variable como encabezado. Si el documento no tiene encabezado, puede que el primer valor se tome como nombre de variable, así que ¡ten cuidado!</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">En este algoritmo es necesario estandarizar los datos, de forma que las variables tendrán el mismo peso para el algoritmo. Sin embargo, no es necesario preocuparse por eso, pues la implementación en DataStudy lo hará en automático.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">Ahora que selecciona tus datos para probar. Después de procesar tus datos, podrás escoger el algoritmo del cuál aprender.</p>', unsafe_allow_html=True)

    docC=None
    docC=st.file_uploader("Cargue su archivo en formato .csv", type=["csv"], key="C")

    if docC is not None:
        #Se cargan los datos
        DataC = pd.read_csv(docC)
        st.write(DataC)

        #Se imprime el mapa de calor de los datos originales
        fig, ax = plt.subplots(figsize=(15, 8))
        CorrData=DataC.corr(method='pearson')
        HeatMp = np.triu(CorrData)
        g1 = sb.heatmap(DataC.corr(), cmap="RdBu_r", annot=True, mask=HeatMp, ax=ax)
        st.pyplot(fig)

        #Se pide al usuario realizar la selección de variables
        varSelect=st.multiselect("Seleccione variables a considerar", DataC.columns.values)

        if len(varSelect)>=1:
            #Se eliminan las variables seleccionadas
            NewData=np.array(DataC[varSelect])
            st.write(NewData)

            #Se estandarizan los datos
            estandarizar = StandardScaler()  # Se instancia el objeto StandardScaler o MinMaxScaler
            MEstandarizada = estandarizar.fit_transform(NewData)  # Se calculan la media y desviación y se escalan los datos
            st.write("Los datos se han estandarizado.")

            #Se selecciona el clustering a realizar
            st.markdown("# Selección de clustering")
            st.markdown('<p align="justify", style="font-size: 20px;">Ahora que los datos están listos, selecciona el clustering que quieras utilizar. Cada uno de ellos se explicará cuando lo selecciones.</p>', unsafe_allow_html=True)
            clustering=st.selectbox("Seleccione un clustering", ["-","Jerárquico", "Particional"])
            
            if clustering=="Jerárquico":
                st.markdown("## Clustering Jerárquico")
                st.markdown('<p align="justify", style="font-size: 20px;">El clustering jerárquico organiza los elementos a partir de una jerarquía en forma de árbol. El árbol representa las relaciones de similitud entre los elementos. En este caso se utiliza un algoritmo llamado algoritmo Ascendente Jerárquico, que consiste en hacer un árbol con todos los elementos como hoja y agrupar los más cercanos hasta obtener el nivel deseado. Los pasos para este algoritmo son:</p>', unsafe_allow_html=True)
                st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Calcular la matriz de distancias/similitud.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Construir un árbol inicial en el que cada elemento es un clúster o una hoja.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Combinar los dos elementos más cercanos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Actualizar la matriz de distancias/similitud.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5. Repetir los pasos 3 y 4 hasta obtener la cantidad de clústeres determinada.</p>', unsafe_allow_html=True)
                st.markdown('<p align="justify", style="font-size: 20px;">Ahora te presentamos los resultados de este algoritmo con los datos que has proporcionado. El proceso de este algoritmo puede ser algo tardado, especialmente la impresión del árbol que se genera, así que ¡por favor, ten paciencia! </p>', unsafe_allow_html=True)
                
                #Se arma el árbol
                st.markdown("### Árbol")
                fig3,ax3 = plt.subplots(figsize=(10, 7))
                Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'), ax=ax3)
                st.pyplot(fig3)

                #Se seleccionan el número de clusteres
                clusteres=st.slider("Seleccione el número de clusteres a considerar",0,15,4)

                #Se selecciona la métrica de distancia
                metrica=st.selectbox("Seleccione una métrica de distancia", ["Euclidiana", "Chebyshev", "Manhattan"])
                if metrica=="Euclidiana":
                    dis='euclidean'
                elif metrica=="Chebyshev":
                    dis='chebyshev'
                elif metrica=="Manhattan":
                    dis='cityblock'
                else:
                    dis='euclidean'

                #Se crean las etiquetas de los elementos en los clusters
                MJerarquico = AgglomerativeClustering(n_clusters=clusteres, linkage='complete', affinity=dis)
                MJerarquico.fit_predict(MEstandarizada)
                
                #Se etiquetan los datos
                st.markdown("### Datos etiquetados")
                DataC['clusterH']=MJerarquico.labels_
                st.write(DataC)

                #Se cuentan los elementos por clústes
                st.markdown("### Datos por clúster")
                st.write(DataC.groupby(['clusterH'])['clusterH'].count())

                #Se muestran los datos de cada clúster
                st.markdown("### Valores de cada clúster")
                CentroidesH=DataC.groupby(['clusterH'])[varSelect].mean()
                st.write(CentroidesH)

                #Se realiza la gráfica
                fig3, ax3 = plt.subplots(figsize=(10,7))
                ax3.scatter(MEstandarizada[:, 0], MEstandarizada[:, 1], c=MJerarquico.labels_)
                ax3.grid()
                st.pyplot(fig3)
            elif clustering=="Particional":
                st.markdown("## Clustering Particional")
                st.markdown('<p align="justify", style="font-size: 20px;">El clustering particional organiza los elementos a partir de una cantidad k determinada, de forma que se generarán tantos clústeres como el valor de k. Este algoritmo se llama k-means y organiza los elementos en base a su similitud.</p>', unsafe_allow_html=True)
                st.markdown('<p align="justify", style="font-size: 20px;">Para implementar el algoritmo, primero es necesario saber el valor de k. Este valor se obtiene a través del método del codo, el cual calcula el desempeño de análisis de los datos proporcionados con distintos valores de k y lo gráfica. Una vez graficado, el método buscará el valor más prominente para tomarlo como k.</p>', unsafe_allow_html=True)
                st.markdown('<p align="justify", style="font-size: 20px;">Con el valor de k determinado, el algoritmo k-means entra en acción a través de los siguientes pasos:</p>', unsafe_allow_html=True)
                st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Se establecen k centroides, los cuales son elementos aleatorios para generar clústeres. <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Cada elemento es asignado al centroide más cercano a partir de las distancias calculadas.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Ya que cada elemento fue asignado, se actualiza el centroide con base en la media de los elementos asignados al clúster.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Se repiten los pasos 2 y 3 hasta que los centroides ya no cambien.</p>', unsafe_allow_html=True)
                st.markdown('<p align="justify", style="font-size: 20px;">Ahora te presentamos los resultados de este algoritmo con los datos que has proporcionado.</p>', unsafe_allow_html=True)
                
                #se obtiene SSE
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0)
                    km.fit(MEstandarizada)
                    SSE.append(km.inertia_)

                #Se grafica SSE en función de k
                st.markdown("### SSE en función de k")
                plt.figure(figsize=(10, 7))
                plt.plot(range(2, 12), SSE, marker='o')
                plt.xlabel('Cantidad de clusters *k*')
                plt.ylabel('SSE')
                plt.title('Elbow Method')
                st.pyplot(plt.show())

                #Se localiza el codo
                kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
                plt.style.use('ggplot')
                st.pyplot(kl.plot_knee())
                st.write("Codo localizado en k="+str(kl.elbow))

                #Se obtiene el clustering particional
                MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
                MParticional.predict(MEstandarizada)

                #Se etiquetan los datos
                st.markdown("### Datos etiquetados")
                DataC['clusterP'] = MParticional.labels_
                st.write(DataC)

                #Se cuentan los elementos por clústes
                st.markdown("### Datos por clúster")
                st.write(DataC.groupby(['clusterP'])['clusterP'].count())
                
                #Se muestran los datos de cada clúster
                st.markdown("### Valores de cada clúster")
                CentroidesP = DataC.groupby(['clusterP'])[varSelect].mean()
                st.write(CentroidesP)

                #Se grafican de los elementos y los centros de los clusters
                plt.rcParams['figure.figsize'] = (10, 7)
                plt.style.use('ggplot')
                colores=['red', 'blue', 'green', 'yellow', 'cyan']
                asignar=[]
                for row in MParticional.labels_:
                    asignar.append(colores[row])

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(MEstandarizada[:, 0],
                        MEstandarizada[:, 1],
                        MEstandarizada[:, 2], marker='o', c=asignar, s=60)
                ax.scatter(MParticional.cluster_centers_[:, 0],
                        MParticional.cluster_centers_[:, 1],
                        MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
                st.pyplot(fig)
            else:
                st.write("**Seleccione un tipo de clustering para continuar.**")


# plt.figure(facecolor="None") -> fondo de afuera transparente