using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
public class navPlayer : MonoBehaviour
{
    public GameObject enemy;
    private NavMeshAgent agent;
    
    // Start is called before the first frame update
    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        agent.speed = 10f;
            
    }

    // Update is called once per frame
    void Update()
    {
        MoveAwayFromEnemy();
    }
    void MoveAwayFromEnemy()
    {
        if (enemy == null)
            return;

        float distanceToEnemy = Vector3.Distance(transform.position, enemy.transform.position);
        Vector3 directionAwayFromEnemy = (transform.position - enemy.transform.position).normalized;

        // Detect wall proximity
        RaycastHit wallHit; // El RaycastHit nos permite da informacion sobre un estructura
        // Physics.Raycast devuelve True si intersecciona con un Colider, y False en caso contrario
        bool isNearWall = Physics.Raycast(transform.position, directionAwayFromEnemy, out wallHit, 5f);

        //Vector3 lateralMovement;

        // Si el enemigo esta a X distancia o cerca de un muro nos movemos lateralmente
        if (distanceToEnemy < 5f || isNearWall)
        {
            // El Vector3.Cross es el producto de dos vectores
            // Creamos un vector perpendicular al eje Y y a la direccion del enemigo
            //Vector3 lateral1Direction = Vector3.Cross(directionAwayFromEnemy, Vector3.up).normalized;
            Vector3 lateralDirection = Vector3.Cross(directionAwayFromEnemy, Vector3.rigth).normalized;

            // Seleccionamos la direccion de forma aleatoria; supera el umbral Derecha, no supera izquierda
            float lateralOffset = Random.value > 0.5f ? 3f : -3f;
            Vector3 lateralMovement = directionAwayFromEnemy * 3f + lateralDirection * lateralOffset;
        }
        else
        {
            // En caso opuesto nos movemos en la direccion contraria del enemigo
            lateralMovement = directionAwayFromEnemy * 3f;
        }

        // Posicion actual + (Dir contraria enemigo || Dir lateral)
        Vector3 destination = transform.position + lateralMovement;

        // Asegurar que el destino esta en el NavMesh
        NavMeshHit navMeshHit;
        if (NavMesh.SamplePosition(destination, out navMeshHit, 5f, NavMesh.AllAreas))
        {
            agent.SetDestination(navMeshHit.position);
        }
        else
        {
            Debug.LogWarning("No se ha encontrado un  destino dentro del NavMesh");
        }
        // Dibuja una l�nea desde la posici�n actual hasta el destino
        Debug.DrawLine(transform.position, destination, Color.red, 0.5f);
        // Muestra el destino en la consola
        Debug.Log("Destino: " + destination);
    } 

    // Hay alguna forma de obtener los limites del plano de forma sencilla ?
}
