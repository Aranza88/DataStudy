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
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay

def logistica():
    st.markdown("# Clasificación/Regresión logística")

    st.markdown("## ¿Qué es?")
    st.markdown('<p align="justify", style="font-size: 20px;">La clasificación logística, también conocida como regresión logística, consiste en un algoritmo que tiene como objetivo predecir valores binarios. Esta clasificación es parecida a la regresión lineal, mas en este caso se realizan predicciones en base a una curva sigmoide, que es una función logística. La fórmula de esta función se presenta a continuación:</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col2.image("img\CL.png", use_column_width='never', width=400)
    st.markdown('<p align="justify", style="font-size: 20px;">Asimismo, es importante tener en cuenta que habrá dos tipos de variables distintas para el funcionamiento. Las variables predictoras que serán en las que se base para la predicción y la variable clase, que es la variable que almacenará la clasificación. Recuerda que este algoritmo es para valores binarios, por lo que la variable clase solo debe de tener dos valores posibles.</p>', unsafe_allow_html=True)
    
    st.markdown("## Características del documento")
    st.markdown('<p align="justify", style="font-size: 20px;">Los datos a analizar deben de estar en formato csv, con los nombres de las variables como encabezado. Además, es necesario que se tenga una variable para clasificación que sea binaria, pues esta será nuestra variable clase.</p>', unsafe_allow_html=True)

    st.markdown("# Aplicación del algoritmo")
    st.markdown('<p align="justify", style="font-size: 20px;">Ahora es momento de probar el algoritmo con tus propios datos. Recuerda las características necesarias del documento y adelante.</p>', unsafe_allow_html=True)
    docL=None
    docL=st.file_uploader("Cargue su archivo en formato .csv", type=["csv"], key="AD")

    if docL is not None:
        #Se cargan los datos
        DataL = pd.read_csv(docL)
        st.write(DataL)
        
        #Se imprime el mapa de calor de los datos originales
        fig, ax = plt.subplots(figsize=(15, 8))
        CorrData=DataL.corr(method='pearson')
        HeatMp = np.triu(CorrData)
        g1 = sns.heatmap(DataL.corr(), cmap="RdBu_r", annot=True, mask=HeatMp, ax=ax)
        st.pyplot(fig)

        st.markdown("## Selección de variables")
        #Se pide al usuario realizar la selección de variables predictoras
        varPredict=st.multiselect("Seleccione variables predictoras", DataL.columns.values)
        
        #Se pide al usuario realizar la selección de variable clase
        varClase=st.multiselect("Seleccione variable clase (1)", DataL.columns.values)
        
        if len(varPredict)>=1 and len(varClase)==1:
            if type(DataL[varClase].values[1][0])==str:
                    DataL = DataL.replace({'M': 0, 'B': 1})
            
            # Variables predictoras
            st.markdown("### Variables predictoras")
            X=np.array(DataL[varPredict])
            st.dataframe(X)

            # Variable clase
            st.markdown("### Variable clase")
            Y = np.array(DataL[varClase])
            st.dataframe(Y)

            st.markdown("## Clasificación")
            muestra=st.slider("Inserte el tamaño de la muestra", 0, 100, step=1)
            if muestra>0:
                muestra=muestra/100
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = muestra, random_state = 1234, shuffle = True)
                
                #Se entrena el modelo a partir de los datos de entrada
                ClasificacionRL = linear_model.LogisticRegression()
                ClasificacionRL.fit(X_train, Y_train)

                #Predicciones probabilísticas de los datos de prueba
                Probabilidad = ClasificacionRL.predict_proba(X_validation)

                #Clasificación final 
                st.markdown("### Clasificación obtenida")
                Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
                st.write(Y_ClasificacionRL)

                #Se calcula la exactitud promedio de la validación
                acurracy=accuracy_score(Y_validation, Y_ClasificacionRL)
                st.write("La exactitud promedio es de "+str(acurracy))

                st.markdown("## Validación del modelo")
                #Matriz de clasificación
                st.markdown("### Matriz de clasificación")
                ModeloClasificacion = ClasificacionRL.predict(X_validation)
                Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                                ModeloClasificacion, 
                                                rownames=['Reales'], 
                                                colnames=['Clasificación']) 
                st.write(Matriz_Clasificacion)

                #Reporte de la clasificación
                st.markdown("### Reporte de la clasificación")
                st.write("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionRL))
                st.markdown(classification_report(Y_validation, Y_ClasificacionRL))

                #CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation, name="Clasificación logística")
                #st.image(CurvaROC)

                #Ecuación del modelo
                st.markdown("## Ecuación del modelo")
                st.write("Intercept:", ClasificacionRL.intercept_)
                st.write('Coeficientes: \n', ClasificacionRL.coef_) 

                #prueba=pd.DataFrame()
                #for v in varPredict:
                #    value=st.number_input("Indique el valor de la variable "+str(v), 0.00000, step=0.00001,format="%.5f")
                #    if value>0:
                #        prueba[v]=value

                #st.dataframe(prueba)
                #if len(prueba.columns)==len(varPredict):
                #    st.write("La predicción del dato seleccionado es: "+ClasificacionRL.predict(prueba))

# 'Texture', 'Area','Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'