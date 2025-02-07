using System.Collections;
using System.Collections.Generic;
using Unity.AI.Navigation.Editor;
using UnityEngine;
using UnityEngine.AI;

public class NavigationScript : MonoBehaviour
{
    public Transform target;
    private NavMeshAgent agent;

    // Start is called before the first frame update
    void Start()
    {
        agent = GetComponent<NavMeshAgent>();

        agent.speed = 100f;             // Velocidad máxima
        agent.acceleration = 1000f;    // Aceleración instantánea
        agent.angularSpeed = 1000f;    // Giro instantáneo
        agent.stoppingDistance = 0f;   // Sin distancia de frenado
        agent.autoBraking = false;     // Sin frenado automático
    }

    // Update is called once per frame
    void Update()
    {
        MoveTowardsRandomTargets();
    }
    void MoveTowardsRandomTargets()
    {
        agent.destination = target.position;
    }
}
