using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;

public class EnemyMovement
{
    private VisionSensor visionSensor; // Referencia al cono de visión
    private SoundSensor soundSensor;  // Referencia a la esfera de sonido
    private NavMeshAgent agent; // Referencia al NavMeshAgent
    private float baseSpeed; // Velocidad base de movimiento
    // private int currentSearchPointIndex = 0;
    // private float rotationSpeed = 1f;
    private float searchRadius = 4f;
    private LayerMask treasureLayer;
    private Vector3 targetPoint = Vector3.zero;
    public RoutineEnemy routineEnemy;
    public EnemyMovement(VisionSensor visionSensor, SoundSensor soundSensor, NavMeshAgent agent, LayerMask treasureLayer, float baseSpeed)
    {
        this.visionSensor = visionSensor;
        this.soundSensor = soundSensor;
        this.agent = agent;
        this.baseSpeed = baseSpeed;
        this.treasureLayer = treasureLayer;
    }

    // Método para patrullar
    public void Patrol(Transform[] waypoints, ref int currentWaypointIndex)
    {
        if (waypoints.Length == 0) return;


        // Verifica si el enemigo ha llegado al waypoint actual
        if (!agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
        {
            currentWaypointIndex = (currentWaypointIndex + 1) % waypoints.Length;
        }

        // Mueve al enemigo hacia el waypoint actual
        MoveToTarget(waypoints[currentWaypointIndex].position);
    }

    // Método para perseguir al jugador
    public void Chase(Transform player)
    {
        MoveToTarget(player.position);
    }

    // Método para buscar en una posición
    public void Search(Vector3 searchPosition)
    {
        MoveToTarget(searchPosition);

        // Si el enemigo ha llegado al destino, elegir un nuevo punto aleatorio
        if (agent.remainingDistance <= agent.stoppingDistance)
        {
            Vector3 randomPoint = searchPosition + Random.insideUnitSphere * searchRadius;
            randomPoint.y = searchPosition.y; // Mantener la misma altura
            MoveToTarget(randomPoint);
        }
    }


    // Método para proteger un tesoro
    public void ProtectTreasure(Vector3 treasurePosition)
    {
        float distanceToTreasure = Vector3.Distance(agent.transform.position, treasurePosition);

        // Si el agente está lejos del tesoro, moverse hacia él
        if (distanceToTreasure > 3)
        {
            MoveToTarget(treasurePosition);
            return;
        }

        if (!IsTreasureNearPosition(treasurePosition, 2f)) 
        {
            List<Vector3> trophyPositions = routineEnemy.GetTrophyPositions();
            trophyPositions.Remove(treasurePosition);
            return;
        }

        // Verifica si el agente ha llegado al punto objetivo o no tiene destino
        float stoppingDistance = 0.5f;
        bool needsNewTarget = targetPoint == Vector3.zero ||
                             Vector3.Distance(agent.transform.position, targetPoint) <= stoppingDistance ||
                             !agent.hasPath ||
                             agent.velocity.magnitude < 0.1f;

        if (needsNewTarget)
        {
            // Definir y validar puntos de patrulla alrededor del tesoro
            Vector3[] patrolPoints = new Vector3[]
            {
            treasurePosition + new Vector3(4, 0, 0),  // Derecha
            treasurePosition + new Vector3(-4, 0, 0), // Izquierda
            treasurePosition + new Vector3(0, 0, 4),  // Adelante
            treasurePosition + new Vector3(0, 0, -4)  // Atrás
            };

            // Intentar hasta 5 veces encontrar un punto válido
            for (int i = 0; i < 5; i++)
            {
                Vector3 potentialTarget = patrolPoints[Random.Range(0, patrolPoints.Length)];

                // Verificar que el punto está en el NavMesh
                NavMeshHit hit;
                if (NavMesh.SamplePosition(potentialTarget, out hit, 2.0f, NavMesh.AllAreas))
                {
                    targetPoint = hit.position;
                    MoveToTarget(targetPoint);
                    Debug.Log("Nuevo punto de patrulla establecido: " + targetPoint);
                    return;
                }
            }

            // Si no encontramos un punto válido, moverse directamente al tesoro
            targetPoint = treasurePosition;
            MoveToTarget(targetPoint);
            Debug.Log("No se encontraron puntos válidos, moviendo al tesoro");
        }
    }



    // TODO: Método para proteger la salida
    public void ProtectExit(GameObject[] exitPoints)
    {
        if (agent.remainingDistance <= agent.stoppingDistance)
        {
            // Elegir un punto de patrulla aleatorio
            Vector3 targetPoint = exitPoints[Random.Range(0, exitPoints.Length)].transform.position;
            // Mover al enemigo al punto de patrulla
            MoveToTarget(targetPoint);
        }
    }

    // Método para moverse a un objetivo
    public void MoveToTarget(Vector3 targetPosition)
    {
        agent.speed = baseSpeed;
        agent.acceleration = 1000f; // Aceleración muy alta
        agent.angularSpeed = 720f;  // Rotación rápida
        agent.stoppingDistance = 0f; // Detenerse exactamente en el destino
        agent.autoBraking = false;  // Desactivar frenado automático
        agent.SetDestination(targetPosition);
    }

    // Método para verificar si el jugador es visible
    public bool IsPlayerVisible()
    {
        Transform player = GameObject.FindGameObjectWithTag("Player")?.transform;
        return player != null && DetectionUtils.IsPlayerVisible(visionSensor, player);
    }

    // Método para verificar si el jugador es audible
    public bool IsPlayerHearable()
    {
        Transform player = GameObject.FindGameObjectWithTag("Player")?.transform;
        return player != null && DetectionUtils.IsPlayerHearable(soundSensor);
    }
    public bool IsTreasureVisible()
    {
        return DetectionUtils.IsTreasureVisible(agent, visionSensor, treasureLayer);
    }
    public bool IsTreasureNearPosition(Vector3 position, float radius)
    {
        return DetectionUtils.IsTreasureNearPosition(visionSensor, position, radius);
    }

}