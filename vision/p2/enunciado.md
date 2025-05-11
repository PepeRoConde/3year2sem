# Contexto
En el campo del análisis de imágenes médicas y, en particular, de las imágenes oftalmológicas, el análisis de la retina presenta una relevancia clínica significativa dado su potencial para el diagnóstico no invasivo de enfermedades oculares y sistémicas. 

La Tomografía de Coherencia Óptica (OCT) es una técnica no invasiva que permite obtener una digitalización volumétrica de alta resolución de los tejidos oculares. Esto convierte a la OCT en una de las técnicas de imagen médica más comunes, ya que permite un estudio rápido y no invasivo de las principales estructuras presentes en la retina. Las imágenes de la OCT proporcionan una visión detallada del estado de las principales estructuras del ojo, como las capas retinianas, lo que permite a los especialistas observar estas estructuras en alta resolución y diagnosticar alteraciones como la presencia de fluido patológico, que pueden estar relacionadas con enfermedades diversas patologías.

Esta práctica se centra en el análisis automático de imágenes OCT para la segmentación de regiones de fluido patológico.

# Material
En el archivo comprimido P2-Material.zip está disponible el siguiente material:
OCT-dataset. Conjunto de datos formado por 50 imágenes OCT con sus correspondientes máscaras binarias.
VCA-p2.ipynb. Código de inicio en un notebook Colab que incluye una Custom PyTorch Dataset class que maneja la carga de las imágenes y máscaras,  definición de arquitectura de red UNet y algunas funciones auxiliares.
# Objetivos
El objetivo de este proyecto es el desarrollo de una metodología para la segmentación automática de las regiones de fluido patológico en imágenes OCT.

- Baseline. Entrenar y evaluar una red base que, dada una imagen OCT proporcione una máscara con las regiones de fluido patológico. 
- Mejoras. Proponer y aplicar posibles mejoras sobre la metodología base de la tarea inicial.
- Experimentos. Los experimentos deben incluir una comparación de las diferentes aproximaciones analizando la influencia de cada una de las mejoras propuestas con respecto a la propuesta base. Deben incorporarse métricas de evaluación apropiadas, como recall, precision, accuracy, IOU, etc. 
- Informe. Redactar un informe que describa y justifique el enfoque propuesto, incluyendo la motivación de los métodos utilizados, la descripción del conjunto de datos y detalles de entrenamiento, los experimentos realizados, así como los resultados extraídos y discusión. El informe debe entregarse en formato PDF. Se recomienda utilizar LaTeX para dar formato al documento (por ejemplo, utilizando una plantilla de conferencia IEEE en la plataforma Overleaf).
 
# Entrega
La fecha de entrega es el 25/05/2025.
Se entregará un archivo comprimido con el código fuente, los modelos entrenados y el informe en pdf a través del espacio habilitado en el Campus Virtual de la asignatura.

Tras la entrega, cada pareja de estudiantes  deberá realizar una defensa de la práctica, donde se evaluará el conocimiento de la misma por parte de cada uno de los miembros del grupo.

# Evaluación
La evaluación de la tarea seguirá las siguientes ponderaciones (con respecto a la puntuación máxima):
- (25%) - Desarrollo de metodología baseline.
- (25%) - Calidad y relevancia de las propuestas de mejora.
- (25%) - Calidad de los experimentos de evaluación.
- (25%) - Calidad del informe final y defensa de la práctica.
