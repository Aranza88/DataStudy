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
# Para generar y almacenar los gráficos dentro del cuaderno
import yfinance as yf
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def arbolesDecision():
    st.markdown("# Árboles de Decisión")

    st.markdown("## ¿Qué son?")
    st.markdown('<p align="justify", style="font-size: 20px;">Los árboles de decisión son muy comunes hoy en día, pues son de os algoritmos más utilizados. En este algoritmo, se busca construir un árbol eficiente y escalable que divida los datos en función de distintas condiciones. Estos árboles pueden ser de pronóstico o clasificación, pero todos ellos cuentan con cuatro elementos principales:</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - <em>Nodo principal:</em> Representa toda la población que se divide en las ramas.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - <em>Nodo de decisión:</em> Todos aquellos nodos que están en niveles intermedios y se encargan de dividir a los datos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - <em>Nodo hoja:</em> Los nodos finales; representan la decisión final.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - <em>Profundidad:</em> El número de niveles que tiene el árbol.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> Adicionalmente, siempre es importante cuidarse del sobreajuste o overfitting, que se presenta cuando un árbol se personaliza demasiado a los datos de prueba. Esto se previene al cuidar la profundidad, es decir, podar el árbol.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> Para la creación de un árbol, podemos describir un algoritmo general, el cual consiste en los siguientes pasos:</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Se crea un nodo raíz con todos los elementos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Se verifica la clase de todos los elementos: si todos los elementos son de la misma clase, el subárbol se cierra, sino, se elige una condición para dividirlos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. El árbol queda subdividido en subárboles.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Se repite el paso 3 hasta cerrar todos los árboles.</p>', unsafe_allow_html=True)

    st.markdown("## Características del documento")
    st.markdown('<p align="justify", style="font-size: 20px;">En este caso, solo se requiere documento para el árbol de clasificación. El documento debe estar en formato csv con los nombres de las variables como encabezado. Asimismo, se debe de tener una variable clasificatoria para utilizarla como variable clase, mientras que las demás cariables serán variables predictoras.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;">En el caso del árbol de pronóstico, lo único necesario es el Ticker de una empresa a analizar. Este dato se puede obtener de Yahoo Finances o, si solo buscas un ejemplo cualquiera, prueba con “AMZN”.</p>', unsafe_allow_html=True)

    #Se selecciona el tipo de bosque a utilizar
    st.markdown("# Selección de tipo de árbol")
    st.markdown('<p align="justify", style="font-size: 20px;">Ahora, selecciona el tipo de árbol que quieras utilizar. Cada uno de ellos se explicará cuando lo selecciones.</p>', unsafe_allow_html=True)
    tipo=st.selectbox("Seleccione tipo de árbol de decisión", ["-","Pronóstico", "Clasificación"])
        
    if tipo=="Pronóstico":
        st.markdown("# Árbol de Decisión: Pronóstico")
        st.markdown('<p align="justify", style="font-size: 20px;">Los árboles de pronóstico o regresión aprenden detalles de los datos de los datos de prueba para hacer predicciones. En este algoritmo se utilizan el error cuadrático medio (MSE) y el error absoluto medio (MAE) como criterios comunes para la predicción. El MSE establece el valor pronosticado de los nodos terminales con respecto al valor medio aprendido, mientras que el MAE establece el valor pronosticado de los nodos terminales con respecto a la mediana. A partir del MSE se obtiene la raíza del error cuadrático medio (RMSE) y se hace una predicción.</p>', unsafe_allow_html=True)

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

                col4, col5, col6 = st.columns(3)
                profundidad=col4.number_input("Indique la profundidad máxima del árbol:", 1, value=1, step=1)
                min_split=col5.number_input("Indique el mínimo de elementos para el split", 1, value=1, step=1)
                min_leaf=col6.number_input("Indique el mínimo de elementos para las hojas", 1, value=1, step=1)

                if min_split>=2:
                    PronosticoAD = DecisionTreeRegressor(max_depth=profundidad, min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=0)
                    PronosticoAD.fit(X_train, Y_train)
                        
                    #Se genera el pronóstico
                    st.markdown("### Pronóstico y valores obtenidos")
                    Y_Pronostico = PronosticoAD.predict(X_test)
                    Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    st.dataframe(Valores)

                    st.write("Exactitud: "+str(r2_score(Y_test, Y_Pronostico)))

                    st.markdown("### Información de los datos obtenidos")
                    st.write('Criterio: \n', PronosticoAD.criterion)
                    st.write('Importancia variables: \n', PronosticoAD.feature_importances_)
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
                    plt.title('Pronóstico')
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plot)

                    st.markdown("### Importancia de las variables")
                    Importancia = pd.DataFrame({'Variable': list(MDatos[['Open', 'High', 'Low']]), 'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.dataframe(Importancia)

                    st.markdown("### Árbol obtenido")
                    plot=plt.figure(figsize=(16,16))  
                    plot_tree(PronosticoAD, feature_names = ['Open', 'High', 'Low'])
                    st.pyplot(plot)

                    #Reporte = export_text(PronosticoAD, feature_names = ['Open', 'High', 'Low'])
                    #st.write(Reporte)
        
    elif tipo=="Clasificación":
        st.markdown("# Árbol de Decisión: Clasificación")
        st.markdown('<p align="justify", style="font-size: 20px;"> Para la clasificación se utilizan básicamente dos conceptos: Entropía y Ganancia de Información. La entropía representa la incertidumbre presenta en la información, mientras que la ganancia representa la cantidad de información que una variable es capaz de aportar. Es así que se sigue el siguiente procedimiento:</p>', unsafe_allow_html=True)
        st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Se calcula la entropía para todas las clases y atributos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Se selecciona el mejor atributo basado en la ganancia de información de cada variable.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Se clasifican los elementos conforme a la selección de variables.</p>', unsafe_allow_html=True)

        st.markdown("# Aplicación del algoritmo")
        st.markdown('<p align="justify", style="font-size: 20px;"> ¡Lo sentimos! Por ahora el algoritmo de Árboles de Decisión: Clasificación aún no está completo, por lo que no se puede utilizar. Vuelva en una próxima versión de DataStudy.</p>', unsafe_allow_html=True)
        """docAD=None
        docAD=st.file_uploader("Cargue su archivo en formato .csv", type=["csv"], key="AD")

        if docAD is not None:
            #Se cargan los datos
            DataAD = pd.read_csv(docAD)
            st.write(DataAD)

            #Se imprime el mapa de calor de los datos originales
            fig, ax = plt.subplots(figsize=(15, 8))
            CorrData=DataAD.corr(method='pearson')
            HeatMp = np.triu(CorrData)
            g1 = sb.heatmap(DataAD.corr(), cmap="RdBu_r", annot=True, mask=HeatMp, ax=ax)
            st.pyplot(fig)

            #Se pide al usuario realizar la selección de variables predictoras
            varPredict=st.multiselect("Seleccione variables predictoras", DataAD.columns.values)
            
            #Se pide al usuario realizar la selección de variable clase
            varClase=st.multiselect("Seleccione variable clase (1)", DataAD.columns.values)

            if len(varPredict)>=1 and len(varClase==1):
                # Variables predictoras
                X=np.array(DataAD[varPredict])
                pd.DataFrame(X)

                # Variable clase
                Y = np.array(DataAD[varClase])
                pd.DataFrame(Y)

                # Creación de los modelos
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size = 0.2, random_state = 0, shuffle = True)

                st.write(len(X_train))
                st.write(len(X_validation))

                #Se entrena el modelo a partir de los datos de entrada
                #ClasificacionAD = DecisionTreeClassifier(random_state=0)
                #ClasificacionAD.fit(X_train, Y_train)

                ClasificacionAD = DecisionTreeClassifier(max_depth=20, 
                                                        min_samples_split=4, 
                                                        min_samples_leaf=2,
                                                        random_state=0)
                ClasificacionAD.fit(X_train, Y_train)

                #Clasificación final 
                Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
                print(Y_ClasificacionAD)

                ValoresMod1 = pd.DataFrame(Y_validation, Y_ClasificacionAD)
                ValoresMod1

                accuracy_score(Y_validation, Y_ClasificacionAD)

                #Matriz de clasificación
                ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
                Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                                ModeloClasificacion1, 
                                                rownames=['Actual'], 
                                                colnames=['Clasificación']) 
                Matriz_Clasificacion1

                #Reporte de la clasificación
                print('Criterio: \n', ClasificacionAD.criterion)
                print('Importancia variables: \n', ClasificacionAD.feature_importances_)
                print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionAD))
                print(classification_report(Y_validation, Y_ClasificacionAD))

                ImportanciaMod1 = pd.DataFrame({'Variable': list(Covid[['SEXO','TIPO_PACIENTE','INTUBADO','NEUMONIA','EDAD','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO','RESULTADO_ANTIGENO','CLASIFICACION_FINAL','UCI']]),
                                    'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                ImportanciaMod1

                plt.figure(figsize=(16,16))  
                plot_tree(ClasificacionAD, feature_names = ['SEXO','TIPO_PACIENTE','INTUBADO','NEUMONIA','EDAD','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO','RESULTADO_ANTIGENO','CLASIFICACION_FINAL','UCI'])
                plt.show()
                
                Reporte = export_text(ClasificacionAD, feature_names = ['SEXO','TIPO_PACIENTE','INTUBADO','NEUMONIA','EDAD','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO','RESULTADO_ANTIGENO','CLASIFICACION_FINAL','UCI'])
                print(Reporte)

                print("Árbol de decisión:", accuracy_score(Y_validation, Y_ClasificacionAD))"""
    else:
        st.write("**Seleccione tipo de árbol de decisión para continuar.**")

    
