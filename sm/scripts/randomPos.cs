using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomPos : MonoBehaviour
{
    public float timer = 0.0f; // Para controlar el tiempo
    public float changeInterval = 5f; // Intervalo para mover el objeto
    public BoxCollider boxCollider; // Asignar el Box Collider en el Inspector

    private Vector3 boundsMin; // Mínimos del collider
    private Vector3 boundsMax; // Máximos del collider

    void Start()
    {
        // Calcula y guarda los límites del Box Collider al inicio
        Bounds bounds = boxCollider.bounds;
        boundsMin = bounds.min;
        boundsMax = bounds.max;
    }

    void Update()
    {
        // Incrementa el temporizador con el tiempo que ha pasado desde el último frame
        timer += Time.deltaTime;

        // Si el temporizador supera el intervalo, cambia la posición y reinicia el temporizador
        if (timer >= changeInterval)
        {
            // Generar una posición aleatoria dentro del área del Box Collider
            Vector3 randomPosition = new Vector3(
                Random.Range(boundsMin.x, boundsMax.x), // Límite en X
                transform.position.y,                   // Mantener Y fijo
                Random.Range(boundsMin.z, boundsMax.z)  // Límite en Z
            );

            // Mueve el objeto a esta posición
            transform.position = randomPosition;

            // Reinicia el temporizador
            timer = 0.0f;
        }
    }
}
