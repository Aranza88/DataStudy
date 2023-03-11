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
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline
import yfinance as yf
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def bosquesAleatorios():
    st.markdown("# Bosques aleatorios")
    
    st.markdown("## ¿Qué son?")
    st.markdown('<p align="justify", style="font-size: 20px;">Los bosques aleatorios se generan como una medida en contra del sobreajuste (aprendizaje demasiado personalizado para los datos de entrenamiento). Estos bosques se conforman de un conjunto de árboles de decisión, ya sean de pronóstico o de clasificación, que se entrenan con distintas muestras para tener realimentación y resultados más precisos. Cabe destacar que todos los árboles de decisión de un bosque aleatorio de pronóstico serán de pronóstico y todos los árboles de un bosque de clasificación serán de clasificación.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">Para generar un bosque, se sigue el siguiente procedimiento:</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Se seleccionan muestras aleatorias a partir del conjunto de datos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Se construye un árbol de decisión para cada muestra.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Se llega a un resultado conjunto, dependiendo del tipo de bosque.</p>', unsafe_allow_html=True)

    st.markdown("## Características del documento")
    st.markdown('<p align="justify", style="font-size: 20px;">En este caso, solo se requiere documento para el árbol de clasificación. El documento debe estar en formato csv con los nombres de las variables como encabezado. Asimismo, se debe de tener una variable clasificatoria para utilizarla como variable clase, mientras que las demás cariables serán variables predictoras.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">En el caso del árbol de pronóstico, lo único necesario es el Ticker de una empresa a analizar. Este dato se puede obtener de Yahoo Finances o, si solo buscas un ejemplo cualquiera, prueba con “AMZN”.</p>', unsafe_allow_html=True)

    #Se selecciona el tipo de bosque a utilizar
    st.markdown("# Selección de tipo de bosque")
    st.markdown('<p align="justify", style="font-size: 20px;">Ahora, selecciona el tipo de bosque que quieras utilizar. Cada uno de ellos se explicará cuando lo selecciones.</p>', unsafe_allow_html=True)
    tipo=st.selectbox("Seleccione tipo de bosque aleatorio", ["-","Pronóstico", "Clasificación"])
        
    if tipo=="Pronóstico":
        st.markdown("# Bosque aleatorio: Pronóstico")
        st.markdown('<p align="justify", style="font-size: 20px;">El bosque aleatorio de pronóstico se utiliza para variables numéricas de valor continuo y consiste en un conjunto de árboles de decisión de pronóstico (para aprender más sobre árboles de decisión, puedes ir a la página de este algoritmo con la barra lateral de la aplicación). En este tipo de bosque, se utiliza el promedio de los resultados de todos los árboles para llegar a un resultado conjunto y determinarlo como decisión final.</p>', unsafe_allow_html=True)

        st.markdown("# Aplicación del algoritmo")
        Ticker=st.text_input("Inserte el Ticker de la compañía que quiere pronosticar:")
        
        if Ticker!="":
            DataAD=yf.Ticker(Ticker)
            col1, col2, col3 = st.columns(3)
            start=col1.date_input("Inserte fecha inicial")
            end=col2.date_input("Inserte fecha final")
            intervalo=str(col3.number_input("Indique el intervalo en días:", 1, value=1, step=1))+'d'

            st.markdown("### Historial obtenido")
            Hist = DataAD.history(start = start, end = end, interval=intervalo)
            st.dataframe(Hist.describe())

            plot=plt.figure(figsize=(20, 5))
            plt.plot(Hist['Open'], color='purple', marker='+', label='Open')
            plt.plot(Hist['High'], color='blue', marker='+', label='High')
            plt.plot(Hist['Low'], color='orange', marker='+', label='Low')
            plt.plot(Hist['Close'], color='green', marker='+', label='Close')
            plt.xlabel('Fecha')
            plt.ylabel('Precio de las acciones')
            plt.grid(True)
            plt.legend()
            st.pyplot(plot)

            st.markdown("### Modificación de columnas y eliminación de nulos")
            MDatos = Hist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
            # En caso de tener valores nulos
            MDatos = MDatos.dropna()
            st.dataframe(MDatos)

            st.markdown("## Generación del modelo")
            
            # Variables predictoras
            st.markdown("### Variables predictoras")
            X = np.array(MDatos[['Open', 'High', 'Low']])
            st.dataframe(X)

            # Variable clase
            st.markdown("### Variable clase")
            Y = np.array(MDatos[['Close']])
            st.dataframe(Y)

            muestra=st.slider("Inserte el tamaño de la muestra", 0, 100, step=1)
            if muestra>0:
                muestra=muestra/100
                
                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = muestra, random_state = 0, shuffle = True)

                col4, col5 = st.columns(2)
                estimadores=col4.number_input("Indique el número de estimadores a considerar:", 1, value=1, step=1)
                profundidad=col5.number_input("Indique la profundidad máxima del árbol:", 1, value=1, step=1)
                col6, col7 = st.columns(2)
                min_split=col6.number_input("Indique el mínimo de elementos para el split", 1, value=1, step=1)
                min_leaf=col7.number_input("Indique el mínimo de elementos para las hojas", 1, value=1, step=1)

                if min_split>=2:
                    #Se genera el modelo
                    PronosticoBA = RandomForestRegressor(n_estimators=estimadores, max_depth=profundidad, min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=0)
                    PronosticoBA.fit(X_train, Y_train)

                    #Se genera el pronóstico
                    st.markdown("### Pronóstico y valores obtenidos")
                    Y_Pronostico = PronosticoBA.predict(X_test)
                    Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    st.dataframe(Valores)

                    st.write("Exactitud: "+str(r2_score(Y_test, Y_Pronostico)))

                    st.markdown("### Información de los datos obtenidos")
                    st.write('Criterio: \n', PronosticoBA.criterion)
                    st.write('Importancia variables: \n', PronosticoBA.feature_importances_)
                    st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                    st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                    st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                    st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))

                    st.markdown("### Confirmación del modelo")
                    plot=plt.figure(figsize=(20, 5))
                    plt.plot(Y_test, color='red', marker='+', label='Real')
                    plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
                    plt.xlabel('Fecha')
                    plt.ylabel('Precio de las acciones')
                    plt.title('Pronóstico de las acciones de Amazon')
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plot)

                    st.markdown("### Importancia de las variables")
                    Importancia = pd.DataFrame({'Variable': list(MDatos[['Open', 'High', 'Low']]),
                            'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.dataframe(Importancia)

                    st.markdown("### Árbol obtenido")
                    Estimador = PronosticoBA.estimators_[50]
                    plot=plt.figure(figsize=(16,16))  
                    plot_tree(Estimador, feature_names = ['Open', 'High', 'Low'])
                    st.pyplot(plot)
    elif tipo=="Clasificación":
        st.markdown("# Bosque aleatorio: Clasificación")
        st.markdown('<p align="justify", style="font-size: 20px;">El bosque aleatorio de pronóstico se utiliza para variables numéricas de valor continuo y consiste en un conjunto de árboles de decisión de pronóstico (para aprender más sobre árboles de decisión, puedes ir a la página de este algoritmo con la barra lateral de la aplicación). En este tipo de bosque, se utiliza un método llamado soft voting, en el que cada resultado de cada árbol se considera como un voto y la decisión final será aquella que cuente con mayoría de votos.</p>', unsafe_allow_html=True)

        st.markdown("# Aplicación del algoritmo")
        st.markdown('<p align="justify", style="font-size: 20px;"> ¡Lo sentimos! Por ahora el algoritmo de Bosques Aleatorios: Clasificación aún no está completo, por lo que no se puede utilizar. Vuelva en una próxima versión de DataStudy.</p>', unsafe_allow_html=True)
        """"
        docBA=None
        docBA=st.file_uploader("Cargue su archivo en formato .csv", type=["csv"], key="BA")

        if docBA is not None:
            #Se cargan los datos
            DataBA = pd.read_csv(docBA)
            st.write(DataBA)

            #Se imprime el mapa de calor de los datos originales
            fig, ax = plt.subplots(figsize=(15, 8))
            CorrData=DataBA.corr(method='pearson')
            HeatMp = np.triu(CorrData)
            g1 = sb.heatmap(DataBA.corr(), cmap="RdBu_r", annot=True, mask=HeatMp, ax=ax)
            st.pyplot(fig)

            #Se pide al usuario realizar la selección de variables predictoras
            varPredict=st.multiselect("Seleccione variables predictoras", DataBA.columns.values)
            
            #Se pide al usuario realizar la selección de variable clase
            varClase=st.multiselect("Seleccione variable clase (1)", DataBA.columns.values)

            if len(varPredict)>=1 and len(varClase==1):
                # Variables predictoras
                x=np.array(DataBA[varPredict])
                pd.DataFrame(x)

                # Variable clase
                y = np.array(DataBA[varClase])
                pd.DataFrame(y)

                # Creación de los modelos
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)

                st.write(len(X_train))
                st.write(len(X_validation))

                ClasificacionBA = RandomForestClassifier(random_state=0)
                ClasificacionBA.fit(X_train, Y_train)

                #ClasificacionBA = RandomForestClassifier(n_estimators=100,
                #                                         max_depth=11, 
                #                                         min_samples_split=4, 
                #                                         min_samples_leaf=2, 
                #                                         random_state=0)
                #ClasificacionBA.fit(X_train, Y_train)

                #Clasificación final 
                Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
                print(Y_ClasificacionBA)

                ValoresMod2 = pd.DataFrame(Y_validation, Y_ClasificacionBA)
                ValoresMod2

                accuracy_score(Y_validation, Y_ClasificacionBA)

                #Matriz de clasificación
                ModeloClasificacion2 = ClasificacionBA.predict(X_validation)
                Matriz_Clasificacion2 = pd.crosstab(Y_validation.ravel(),
                                                    ModeloClasificacion2,
                                                    rownames=['Reales'],
                                                    colnames=['Clasificación']) 
                Matriz_Clasificacion2

                #Reporte de la clasificación
                print('Criterio: \n', ClasificacionBA.criterion)
                print('Importancia variables: \n', ClasificacionBA.feature_importances_)
                print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionBA))
                print(classification_report(Y_validation, Y_ClasificacionBA))

                Importancia2 = pd.DataFrame({'Variable': list(Covid[['SEXO','TIPO_PACIENTE','INTUBADO','NEUMONIA','EDAD','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO','RESULTADO_ANTIGENO','CLASIFICACION_FINAL','UCI']]), 
                                'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
                Importancia2

                print("Bosque aleatorio:", accuracy_score(Y_validation, Y_ClasificacionBA))
    """
    else:
        st.write("**Seleccione tipo de árbol de decisión para continuar.**")

