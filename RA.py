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
import time
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori

def reglasAsociacion():
    st.markdown("# Reglas de asociación")

    st.markdown("## ¿Qué son?")
    st.markdown('<p align="justify", style="font-size: 20px;">Las reglas de asociación consisten en un algoritmo basado en reglas que se utiliza para encontrar relaciones entre los datos. Esto funciona de forma que se identifican patrones que se encuentran en un conjunto de datos para predecir secuencias. Este tipo de algoritmos se puede encontrar en sistemas de recomendación y se basan en qué tan frecuentes son ciertas combinaciones.</p>', unsafe_allow_html=True)

    st.markdown("## ¿Cómo funciona?")
    st.markdown('<p align="justify", style="font-size: 20px;">Este algoritmo se puede dividir en distintos pasos: </p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Establecer un soporte mínimo de ocurrencias. Este número se refiere a el mínimo para considerar una combinación como regla<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Generar una lista de un ítem y seleccionar aquellas ocurrencias que cumplan con el soporte mínimo. Por ejemplo, si e soporte mínimo es 2, los ítems que no se repitan al menos dos veces en los datos no serán seleccionados.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Utilizar la lista de un ítem para generar una lista de dos ítems que cumplan el soporte mínimo. Esto quiere decir que la nueva lista verificará las ocurrencias de dos ítems a la vez, es decir, ¿cuántas veces aparecen los dos ítems en una sola transacción? Después, se seleccionan aquellas combinaciones que cumplan con el soporte mínimo.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Utilizar la lista para generar una lista de tres ítems, luego la de tres para cuatro y así sucesivamente hasta construir un conjunto del total de ítems disponibles. Es así que las combinaciones se verificarán una por una conforme al soporte mínimo y se establecerán reglas con las combinaciones que cumplan. </p>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<p align="justify", style="font-size: 20px;">Este procedimiento es el funcionamiento del algoritmo en sí, pero se deben de tener en cuenta tres variables para utilizar la implementación de DataStudy.</p>', unsafe_allow_html=True)
    st.markdown('<p align="justify", style="font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -	Soporte: Ya indicamos lo importante que llega a ser. Al momento de implementar el algoritmo, esta variable indica cuán importante es una regla dentro del total de datos.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -	Confianza: Indica que tan fiable es una regla. Este valor se da en porcentaje y el algoritmo no funcionará mientras este valor sea 0.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -	Lift o elevación: Este valor indica el nivel de relación entre el antecedente y consecuente de la regla.</p>', unsafe_allow_html=True)

    st.markdown("## Características del documento")
    st.markdown('<p align="justify", style="font-size: 20px;">Para utilizar este algoritmo, es necesario que los datos sean TRANSACCIONALES, es decir, que los datos sean operativos, que se empleen para controlar o ejecutar tareas y que estén altamente normalizados y se almacenen en tablas. Para utilizar el algoritmo en DataStudy, es necesario cerciorarse también de que los datos no tengan encabezado y estén en un documento csv, donde cada objeto de la transacción se encuentre en una diferente casilla. Si los datos están divididos por comas, ¡por favor busca otro dataset o cambia la organización para que el algoritmo implementado en DataStudy funcione correctamente!</p>', unsafe_allow_html=True)

    st.markdown("# Prueba el algoritmo")
    st.markdown('<p align="justify", style="font-size: 20px;">Ahora es momento de que pruebes el algoritmo con tu propio archivo. Recuerda las caracterísitcas necesarias del documento y adelante.</p>', unsafe_allow_html=True)

    docRA=None
    docRA = st.file_uploader("Cargue su archivo en formato .csv", type=["csv"], key="RA")

    if docRA is not None:
        # Se cargan los datos
        st.markdown("### Datos insertados")
        DataRA=pd.read_csv(docRA, header=None)
                
        #Se incluyen todas las transacciones en una sola lista
        Transacciones = DataRA.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
        
        #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 1 #Basura para que no quede vacía

        #Se agrupa los elementos
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje -> 0.06=6%
        ListaM = ListaM.rename(columns={0 : 'Item'}) #Renombrar la columna en la posición 0
        st.dataframe(ListaM)

        # Se genera un gráfico de barras
        plot=plt.figure(figsize=(16,20), dpi=300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='cyan')
        st.pyplot(plot)

        #st.header("Preparación de datos")
        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice
        Lista = DataRA.stack().groupby(level=0).apply(list).tolist()

        st.markdown("### Aplicación del algoritmo")
        
        #Se insertan los datos para la configuración
        c1, c2= st.columns(2) #Para poner dos inputs en una misma línea
        with st.container():
            support=c1.number_input("Inserte el soporte mínimo", 0.001, 1.0, step=0.001, format="%.3f")
            lift=c2.number_input("Inserte la elevación", 1.1, 10.0, step=0.1,format="%.1f")
        confidence=st.slider("Inserte la confianza en porcentaje", 0, 100, step=1)
        
        if confidence>0:
            confidence=confidence/100
                        
            reglas = apriori(Lista, min_support=support, min_confidence=confidence, min_lift=lift)
            resApriori = list(reglas)
            res=pd.DataFrame(resApriori)
            st.dataframe(res)
            
            st.markdown('<p align="justify", style="font-size: 20px;">Ahora que están los resultados, puedes elegir una regla que quieras observar más a detalle. Para esto utiliza el siguiente slider:</p>', unsafe_allow_html=True)

            rule=st.slider("Seleccione una regla a observar", 0, int((res.size/3)-1), step=1)

            st.markdown("##### Regla "+str(rule)+":")
            items=""
            for i in resApriori[rule].items: 
                items=items+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- "+str(i)+"  \n"
            st.markdown("Items:  \n"+items)
            st.markdown("Soporte: "+str(resApriori[rule].support))
            st.markdown("Confianza: "+str(resApriori[rule].ordered_statistics[0].confidence))
            st.markdown("Elevación: "+str(resApriori[rule].ordered_statistics[0].lift))

            