using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.AI;
using static EnemyMovement;

public class RoutineEnemy : MonoBehaviour
{
    // Referencia al cono de visi�n
    private VisionSensor fieldOfView;
    // Ajustes del cono de visi�n
    [SerializeField] private float fieldOfViewAngle = 90f;
    [SerializeField] private float viewDistance = 5f;
    [SerializeField] private float speed = 5f;
    [SerializeField] private Color visionConeColor = new Color(1f, 0f, 0f, 0.3f);
    [SerializeField] private LayerMask layerObstacles;
    [SerializeField] private LayerMask treasureLayer;


    // Referencia a la esfera de sonido
    private SoundSensor soundSphere;
    // Ajustes de la esfera de sonido
    [SerializeField] private float soundRadius = 8f;

    private EnemyMovement movement; // Referencia a la l�gica de seguimiento
    private NavMeshAgent agent; // Referencia al NavMeshAgent
    public GameManager gameManager;
    [SerializeField] private List<Vector3> trophyPositions;


    // Estado actual
    public enum EnemyState
    {
        Patrol,
        Chase,
        Search,
        ProtectTreasure,
        Alarm,
        ProtectExit
    }
    public EnemyState currentState = EnemyState.Patrol;

    // Waypoints para patrulla
    public Transform[] waypoints;
    private int currentWaypointIndex = 0;

    // �ltima posici�n conocida del jugador
    private Vector3 lastKnownPlayerPosition;
    // El tiempo es desde que lo pierde, no desde que llega a la posicion
    [SerializeField] private int maxSearchTime = 15;
    private float currentSearchTime = 0f;
    private GameObject[] exitPoints;

    void Awake()
    {
        // Configurar visi�n y sonido
        var settingsVision = new VisionConeSettings(
            fov: fieldOfViewAngle,
            viewDistance: viewDistance,
            color: visionConeColor,
            obstaclesMask: layerObstacles,
            trophyLayer: treasureLayer
        );
        var settingsSound = new SoundSphereSettings(radius: soundRadius);
        fieldOfView = VisionConeFactory.CreateVisionCone(transform, settingsVision);
        soundSphere = SoundSphereFactory.CreateSoundSphere(transform, settingsSound);

        // Obtener referencias
        agent = GetComponent<NavMeshAgent>();
        movement = new EnemyMovement(fieldOfView, soundSphere, agent, treasureLayer, speed);
        // L�gica para proteger la salida
        exitPoints = GameObject.FindGameObjectsWithTag("EscapePoint");
    }

    void Update()
    {
        // Actualizar la direcci�n del cono de visi�n
        if (fieldOfView != null)
        {
            fieldOfView.SetViewDirection(transform.forward);
        }

        // Gestionar el estado actual
        switch (currentState)
        {
            case EnemyState.Patrol:
                HandlePatrol();
                break;
            case EnemyState.Chase:
                HandleChase();
                break;
            case EnemyState.Search:
                HandleSearch();
                break;
            case EnemyState.ProtectTreasure:
                HandleProtectTreasure();
                break;
            case EnemyState.Alarm:
                HandleAlarm();
                break;
            case EnemyState.ProtectExit:
                HandleProtectExit();
                break;
        }
    }
    // A�ade estas variables a nivel de clase
    private bool treasureHasBeenSeen = false;      // �Ha visto el tesoro alguna vez?
    private bool treasureIsCurrentlyVisible = false;   // �Ve el tesoro ahora?
    private Vector3 lastKnownTreasurePosition;     // �ltima posici�n conocida del tesoro
    private float treasureCheckCooldown = 2f;      // Tiempo entre verificaciones de desaparici�n
    private float lastTreasureCheckTime = 0f;      // �ltimo momento en que se verific�
    private bool treasureConfirmedMissing = false; // Confirmaci�n de que el tesoro ha desaparecido

    private void HandlePatrol()
    {
        // Si el jugador es detectado, cambiar a persecuci�n
        if (movement.IsPlayerVisible() || movement.IsPlayerHearable())
        {
            currentState = EnemyState.Chase;
            return;
        }

        // Actualizar la visibilidad actual del tesoro
        treasureIsCurrentlyVisible = movement.IsTreasureVisible();

        // Si vemos el tesoro, actualizar su posici�n conocida
        if (treasureIsCurrentlyVisible)
        {
            treasureHasBeenSeen = true;
            lastKnownTreasurePosition = GetVisibleTreasurePosition(); // Necesitar�s implementar esta funci�n
            lastTreasureCheckTime = Time.time; // Actualizar el tiempo de �ltima verificaci�n
            treasureConfirmedMissing = false;  // Resetear la confirmaci�n si vemos el tesoro
            // Debug.Log("Tesoro visible en posici�n: " + lastKnownTreasurePosition);
        }

        // Verificar si el tesoro ha desaparecido solo si:
        // 1. Lo hemos visto antes
        // 2. No lo vemos ahora
        // 3. Estamos lo suficientemente cerca de su �ltima posici�n conocida
        // 4. Ha pasado suficiente tiempo desde la �ltima verificaci�n
        if (treasureHasBeenSeen && !treasureIsCurrentlyVisible && !treasureConfirmedMissing)
        {
            float distanceToLastKnownPosition = Vector3.Distance(transform.position, lastKnownTreasurePosition);

            // Si estamos cerca de donde deber�a estar el tesoro y ha pasado el tiempo suficiente
            if (distanceToLastKnownPosition < fieldOfView.viewDistance &&
                Time.time - lastTreasureCheckTime > treasureCheckCooldown)
            {
                // Verificar expl�citamente que no hay tesoro cerca de la �ltima posici�n conocida
                if (!movement.IsTreasureNearPosition(lastKnownTreasurePosition, 2f)) // 2f es un radio de b�squeda
                {
                    treasureConfirmedMissing = true;
                    Debug.Log("�ALERTA! El tesoro ha desaparecido de su posici�n: " + lastKnownTreasurePosition);
                    // Aqu� implementas la reacci�n del agente: alarma, b�squeda, etc.
                    currentState = EnemyState.ProtectTreasure;
                }

                // Actualizar el tiempo de verificaci�n independientemente del resultado
                lastTreasureCheckTime = Time.time;
            }
        }

        // Mover al enemigo al siguiente waypoint
        movement.Patrol(waypoints, ref currentWaypointIndex);
    }
    private void HandleChase()
    {
        Transform player = GameObject.FindGameObjectWithTag("Player")?.transform;
        movement.Chase(player);

        // Guardar la �ltima posici�n conocida del jugador
        if (player != null)
        {
            lastKnownPlayerPosition = player.position;
        }

        // Transici�n a Search si el jugador ya no es detectado
        if (!movement.IsPlayerVisible() && !movement.IsPlayerHearable())
        {
            movement.MoveToTarget(lastKnownPlayerPosition);
            currentState = EnemyState.Search;
        }
    }
    private void HandleSearch()
    {
        Debug.Log(currentSearchTime);
        currentSearchTime += Time.deltaTime;
        // Verificar si el enemigo ha llegado a la �ltima posici�n conocida
        if (agent.remainingDistance <= agent.stoppingDistance)
        {
            // Realizar la b�squeda activa (girar en el lugar, moverse a puntos cercanos, etc.)
            movement.Search(lastKnownPlayerPosition);
        }
        else if (agent.pathPending)
        {
            // Si el enemigo a�n no ha llegado, reiniciar el temporizador
            currentSearchTime = 0;
        }

        // Verificar si el jugador es detectado durante la b�squeda
        if (movement.IsPlayerVisible() || movement.IsPlayerHearable())
        {
            currentState = EnemyState.Chase;
            currentSearchTime = 0; // Reiniciar el temporizador
            return;
        }

        // Si se completa el tiempo de b�squeda
        if (currentSearchTime >= maxSearchTime)
        {
            if (gameManager.collectedTrophies < gameManager.totalTrophies)
            {
                currentState = EnemyState.ProtectTreasure;
            }
            else
            {
                /*
                 * ESTO DE AQUI SERIA UNA PARTE DE COMUNICACION PERO LA HACEMOS MAL APOSTA
                 * - LO LOGICO SERIA QUE AL DAR LA ALARMA SE AVISASE A TODOS
                 * - PERO SOLO VAMOS A CERRAR LAS PUERTAS PARA EL JUGADOR Y NADA MAS,
                 * ENTONCES AL ENTRAR A ESTE ESTADO IRIA A LA SALA Y DARIA LA ALARMA AUNQUE YA ESTE DADA
                 */
                // Si el jugador tiene todos los trofeos, proteger la salida
                currentState = EnemyState.Alarm;
                // currentState = EnemyState.ProtectExit;
                // CloseAllDoors(); // Cerrar puertas
            }

            currentSearchTime = 0; // Reiniciar el temporizador
        }
    }

    private void HandleAlarm() { /* COMPLETAR CUANDO TENGAMOS LAS PUERTAS CREADAS */ }
    private void HandleProtectTreasure()
    {
        // Si el jugador es detectado, cambiar al estado de persecuci�n
        if (movement.IsPlayerVisible() || movement.IsPlayerHearable())
        {
            currentState = EnemyState.Chase;
            return; // Salir del m�todo para evitar ejecutar el resto del c�digo
        }

        // Obtener el trofeo m�s cercano
        Vector3 nextTrophyPosition = GetNextTrophyPosition();

        // Verificar si el trofeo existe
        if (nextTrophyPosition != null)
        {
            // Proteger el tesoro (moverse hacia el trofeo)
            movement.ProtectTreasure(nextTrophyPosition);
        }
        else
        {
            // Si no hay trofeos, cambiar al estado de proteger la salida
            currentState = EnemyState.ProtectExit;
        }
    }

    public List<Vector3> GetTrophyPositions()
    {
        return trophyPositions;
    }
    private Vector3 GetNextTrophyPosition()
    {
        return trophyPositions[0];
    }

    private void HandleProtectExit()
    {
        if (movement.IsPlayerVisible() || movement.IsPlayerHearable())
        {
            currentState = EnemyState.Chase;
        }

        movement.ProtectExit(exitPoints);
    }

    // M�todo para obtener la posici�n del tesoro visible
    private Vector3 GetVisibleTreasurePosition()
    {
        // Esta funci�n deber�a devolver la posici�n del tesoro que est� viendo actualmente
        Collider[] treasuresInView = Physics.OverlapSphere(
            transform.position,
            fieldOfView.viewDistance,
            fieldOfView.LayerTreasures
        );

        foreach (Collider treasureCollider in treasuresInView)
        {
            // L�gica similar a IsTreasureVisible para verificar �ngulo y obst�culos
            // Si encuentra un tesoro visible, retorna su posici�n
            return treasureCollider.transform.position;
        }

        // Si no encuentra ninguno, retorna la �ltima posici�n conocida o Vector3.zero
        return lastKnownTreasurePosition;
    }
}