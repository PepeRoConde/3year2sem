using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class navEnemy : MonoBehaviour
{
    public GameObject player;
    private NavMeshAgent agent;

    // Start is called before the first frame update
    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        agent.speed = 5f;

    }

    // Update is called once per frame
    void Update()
    {
        MoveToPlayer();
    }
    void MoveToPlayer()
    {
        if (player == null) { return; } // No tenemos enemigo asi que no hacemos nada

        // Asignamos la posición del jugador como destino para que el enemigo lo persiga
        Vector3 destination = player.transform.position;

        // Dibuja una línea desde la posición actual hasta el destino
        Debug.DrawLine(transform.position, destination, Color.blue, 0.5f);
        // Muestra el destino en la consola
        Debug.Log("Destino: " + destination);

        // Asignamos la nueva posicion
        agent.SetDestination(destination);
    }
}
