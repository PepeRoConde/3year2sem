## Contexto

Esta práctica se enmarca en el contexto de una plataforma Smartports que permite la integración y comunicación entre diferentes sistemas para la gestión global e inteligente de infraestructuras portuarias. Entre los sistemas que integra la plataforma, el módulo de videovigilancia para el control de eventos de atraque y desatraque es de crucial importancia dado el volumen de tráfico y operaciones que representa en la operativa diaria de los puertos.

Este módulo está basado en el análisis automático del flujo de vídeo generado por un conjunto de cámaras CCTV (Circuito Cerrado de Televisión) para la detección de los eventos de interés, permitiendo el registro del instante temporal y el envío de notificaciones con la información asociada a cada operación detectada.

En este contexto se identifican dos tareas relevantes para la detección de eventos de interés: la clasificación de imágenes para determinar la presencia o ausencia de algún barco en la escena y la clasificación de imágenes para determinar la presencia o ausencia de algún barco atracado. 

## Material

Se dispone de un dataset formado por 294 imágenes capturadas por diferentes cámaras instaladas en entornos portuarios a diferentes horas del día y con diferentes condiciones de iluminación. Cada imagen tiene una etiqueta asociada a la presencia de barco (1 si hay barco y 0 si no hay barco) y una etiqueta asociada a la presencia de barco atracado (1 atracado y 0 no atracado).

## Objetivos

Práctica realizada por parejas.

 - Tarea 1 Implementación de Dataset class personalizada para la carga del dataset y su uso en tareas de clasificación.
 - Tarea 2 Clasificación Ship/No-ship. Dada una imagen de entrada predecir la presencia o ausencia de barcos en la escena. Partiendo de un modelo de red base:
Entrenamiento y validación desde cero, con y sin data augmentation (2 modelos).
Refinamiento y validación de modelo preentrenado, con y sin data augmentation (2 modelos).
    El aumento de datos debe realizarse considerando las características específicas de este dominio (diferentes punto de vista, escala, condiciones de iluminación, etc.)
 - Tarea 3 Redactar un informe breve en el que se explique el enfoque propuesto indicando los detalles de cada una de las configuraciones, así como la presentación y análisis de los resultados. El informe debe entregarse en formato pdf.
 - Tarea 4 [opcional] Realizar la tarea 2 cambiando el objetivo de clasificación a Docked/Undocked. Dada una imagen de entrada predecir la ausencia o presencia de barcos atracados.
## Entrega
La fecha de entrega es la fijada en la tarea del campus virtual.

Se entregará un archivo comprimido con el código fuente, los modelos entrenados y el informe en pdf a través del espacio habilitado en el Campus Virtual de la asignatura. Los modelos entrenados pueden ser demasiado pesados para el límite de tamaño permitido en el Campus Virtual. En ese caso, los modelos se subirán a OneDrive y se creará un enlace para compartirlos que será incluido en la entrega de la tarea. 

Tras la entrega, cada estudiante deberá realizar una defensa por parejas de la práctica. Durante la defensa se utilizará un conjunto de datos de test no incluido en el dataset inicial, para evaluar la capacidad de generalización de los modelos ante datos no vistos previamente.

## Evaluación
 - Tarea 1 - Creación de dataset personalizado (2 pt)
 - Tarea 2 - Clasificación Ship/No-ship (5 pt)
 - Tarea 3 -  Informe (3 pt)
 - Tarea 4 [opcional] - Clasificación Docked/Undocked (3 pt)
