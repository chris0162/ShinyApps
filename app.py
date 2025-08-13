from shiny import App, render, ui, reactive

# ----------------- Librerias -----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc



# ----------------- Carga del dataset -----------------

url='https://raw.githubusercontent.com/chris0162/Promidat/refs/heads/main/ME06_Tarea2/diagnostico_cancer_mama.csv'
df = pd.read_csv(url,index_col="index")

df_model = df.copy()
# asignacion de variable a predecir y transformacion a numeric
Y = df_model["diagnosis"]
X = df_model.drop(columns=["diagnosis"])  # Ajusta si tu columna objetivo tiene otro nombre



# ----------------- Interfaz de Usuario (UI) -----------------
app_ui = ui.page_fluid(
            ui.div(
                ui.h1('Tarea2_Christian_Araya'),
                ui.p('PROMiDAT Iberoamericano'),
                ui.p('Máster Ciencia Datos e Inteligencia Artificial'),
                ui.p('ME3006 - Programación Avanzada de Dashboards'),
                style="background-color: #e6e6e6; padding: 20px; border-radius: 10px;"
            ),
# ui Tab1, Muestra Dataset  -----------------------------------           
            ui.page_navbar(
            # Cada nav es un tab de navegación.
                ui.nav_panel( "DataSet" , # Etiqueta
                            "Fuente y descripcion del dataset" ,
                            ui.output_data_frame("tabla_datos")),

# ui Tab2, Descripcion de variables ----------------------------           
                ui.nav_panel("Descripcion de variables", # Etiqueta tab2  
                             "Análisis de normalidad de variables",
                            ui.input_select(
                                id = "analizar_variable", # Id de la entrada.
                                label = "Seleccione la variable: ",  # Título de la entrada.
                                # choices = ["a", "b", "c", "d"],  # Opciones posibles.
                                choices = list(df.columns),  # Opciones posibles.
                                selected = "radius_mean",  # Opción seleccionada por defecto.
                                multiple = False,  # Permite varias opciones si es True.
                                width = "30%"  # Ancho de la entrada.
                                ),
                            ui.output_text_verbatim("tipo_variable"),
                            ui.output_plot("hist_plot"),
                            ui.output_text_verbatim("shapiro_output")
                            ), # Contenido
# ui Tab3, Modelo Regresion Logistica -------------------------   
                ui.nav_panel("Modelo Regresion Logistica", # Etiqueta tab3  
                            "A continuacion se muestran los resultados del modelo Regresion \
                            Logistica utiliza para predecir el diagnostico de cancer de mama \
                            a partir de las variables predictoras del dataset. Se utilizo para \
                            el modelo la posibilidad de ajustar el tamaño de la particion de testing \
                            como el umbral para la probabilidad de corte para las predicciones",   
                            ui.div(style="height: 20px;"), # Espacio vertical en blanco   
                            ui.input_slider(
                                id = "valor_umbral", # Id de la entrada.
                                label = "Selector para umbral de probabilidad de corte: ",  # Título de la entrada.
                                min = 0.0,  # Mínimo valor posible.
                                max = 1.0,  # Máximo valor posible.
                                value = 0.5,  # Valor por defecto.
                                step = 0.05,  # Intervalo de incremento o decremento.
                                width = "50%" ), # Ancho de la entrada.
                            ui.input_radio_buttons(
                                id = "valor_testing", # Id de la entrada.
                                label = "Selector tamaño de muestra para testing: ",  # Título de la entrada.
                                choices = ["10%", "15%", "20%", "25%"],  # Opciones posibles.
                                selected = "20%",  # Opción seleccionada por defecto.
                                inline = True,  # Valor por defecto.
                                width = "100%" ), # Ancho de la entrada.
                            ui.output_plot("plot_barras"),
                            ui.output_plot("plot_roc"),
                            ui.output_ui("metricas_texto")
                            ), # Final de nav_panel tab3
                            
# ui Configuracion -------------------------   

                # Titulo
                title = "Opciones", # titulo pestaña navegador
                
                # Color del encabezado
                bg = "steelblue"
                ) # Final page_navbar

) # Final ui.page_fluid

# ----------------- Lógica del Servidor -----------------
def server(input, output, session):

# server Tab1, Muestra Dataset  -----------------------------------           
      @render.data_frame
      def tabla_datos():
           datos_originales = df
           return (datos_originales)
      
# server Tab2, Descripcion de variables ----------------------------           
      analisis_variable = reactive.Value(None)

      @output
      @render.text
      def tipo_variable():
       variable_nombre = input.analizar_variable()
       variable_tipo = str(df[variable_nombre].dtypes)
       res = "Variable: " + variable_nombre + "     Tipo de variable: " + variable_tipo
       return res
      
      @render.plot
      def hist_plot():
       # Crea el gráfico de Matplotlib dentro de la función decorada
       variable_nombre = input.analizar_variable()
       fig, ax = plt.subplots()
       ax.hist(df[variable_nombre], bins=30, edgecolor='black')
       ax.set_xlabel(variable_nombre)
       ax.set_ylabel('Frecuencia')
       ax.set_title('Histograma de ' + variable_nombre)
       return fig
      
      @render.text
      def shapiro_output():
       variable_nombre = input.analizar_variable()
       stat, p = shapiro(df[variable_nombre])
       # Formatear el texto de salida
       output_text = (f"Estadístico de Shapiro-Wilk: {stat:.4f}, p-valor: {p:.4f}\n\n")
       if p > 0.05: 
           output_text += "No se rechaza la hipótesis nula: la variable " + variable_nombre + " parece seguir una distribución normal."
       else:
           output_text += "Se rechaza la hipótesis nula: la variable " + variable_nombre + " no sigue una distribución normal."
       return output_text

# server Tab3, Modelo Regresion Logistica -------------------------   

      def logistica_con_metricas(X, Y, p=0.5, n=0.2):
            """
            Ejecuta regresión logística con escalado de X, conversión de Y a numérico si es categórica,
            aplica un umbral de probabilidad p y partición de test de tamaño n.
            Devuelve la precisión global, precisión por categoría, gráfico de barras y curva ROC.
            
            Parámetros:
            ----------
            X : pandas.DataFrame
                DataFrame con las variables predictoras.
            Y : pandas.Series
                Serie con la variable objetivo.
            p : float, opcional
                Umbral de probabilidad de corte para las predicciones. Por defecto es 0.5.
            n : float, opcional
                Tamaño de la partición de prueba (test_size). Por defecto es 0.2.

            Retorna:
            -------
            tuple
                Una tupla que contiene:
                - precision_global (float): La precisión global del modelo.
                - precision_categorias (dict): Un diccionario con la precisión por categoría.
                - fig_barras (matplotlib.figure.Figure): La figura del gráfico de barras.
                - fig_roc (matplotlib.figure.Figure): La figura de la curva ROC.
            """
            # Escalar X
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Convertir Y a numérico si es categórica
            if Y.dtype == "O" or str(Y.dtype).startswith("category"):
                clases = np.unique(Y)
                if len(clases) == 2:
                    Y_num = Y.map({clases[0]: 0, clases[1]: 1})
                else:
                    Y_num = Y.astype("category").cat.codes
            else:
                Y_num = Y

            # Separar en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_num, test_size=n, random_state=42)
            
            # Entrenar modelo
            modelo = LogisticRegression(max_iter=50)
            modelo.fit(X_train, y_train)
            
            # Probabilidades de la clase positiva
            probs = modelo.predict_proba(X_test)[:, 1]
            
            # Predicción con umbral personalizado
            y_pred_custom = (probs >= p).astype(int)
            
            # Precisión global
            precision_global = accuracy_score(y_test, y_pred_custom)
            
            # Precisión por categoría
            reporte = classification_report(y_test, y_pred_custom, output_dict=True, zero_division=0)
            precision_categorias = {f'Clase {k}': v['precision'] for k, v in reporte.items() if k in ['0', '1']}
            
            # Gráfico de barras
            fig_barras, ax_barras = plt.subplots(figsize=(6,4))
            categorias = ['Global'] + list(precision_categorias.keys())
            valores = [precision_global] + list(precision_categorias.values())

            ax_barras.bar(categorias, valores, color=['steelblue', 'orange', 'green'])
            ax_barras.set_ylim(0, 1)
            ax_barras.set_ylabel('Precisión')
            ax_barras.set_title('Precisión global y por categoría')
            
            # Curva ROC
            fpr, tpr, thresholds = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(6,4))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (área = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('Tasa de Falsos Positivos')
            ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
            ax_roc.set_title('Curva ROC')
            ax_roc.legend(loc="lower right")
                
            # Retornar los objetos de las figuras
            return precision_global, precision_categorias, fig_barras, fig_roc

    # Función reactiva para ejecutar el modelo
      @reactive.Calc
      def resultados_modelo():
        # Limpiar y convertir el valor del radio_button
        test_size_str = input.valor_testing()
        test_size_num = float(test_size_str.strip('%')) / 100

        # Llama a la función del modelo con los inputs reactivos
        return logistica_con_metricas(
            X=X,
            Y=Y,
            p=input.valor_umbral(),
            n=test_size_num
        )

      @render.plot
      def plot_barras():
        # Usa los resultados de la función reactiva
         _, _, fig_barras, _ = resultados_modelo()
         return fig_barras

      @render.plot
      def plot_roc():
        # Usa los resultados de la función reactiva
         _, _, _, fig_roc = resultados_modelo()
         return fig_roc

      @render.ui
      def metricas_texto():
        # Usa los resultados de la función reactiva
        precision_global, precision_categorias, _, _ = resultados_modelo()
        return ui.div(
            ui.h4("Métricas del Modelo"),
            ui.p(f"Precisión Global: {precision_global:.2f}"),
            ui.p(f"Precisión Clase 0 (Benigno): {precision_categorias.get('Clase 0', 0):.2f}"),
            ui.p(f"Precisión Clase 1 (Maligno): {precision_categorias.get('Clase 1', 0):.2f}")
        )




# ----------------- Inicialización de la Aplicación -----------------
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()






