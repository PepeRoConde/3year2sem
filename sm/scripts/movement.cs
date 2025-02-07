using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movement : MonoBehaviour
{
    public GameObject targetObject1;
    public GameObject targetObject2;
    private bool visited1 = false; // Variable para alternar entre objetivos

    void Start()
    {
        Transform trans = GetComponent<Transform>();
    }

    void Update()
    {
        Vector3 pos = transform.position;
        // transform.position = new Vector3(pos.x + 0.01f, pos.y, pos.z);
        MoveTowardsTargets();
    }

    // Se podria hacer para "X" estados, utilizando un dict/lista que contenga el siguiente estado
    void MoveTowardsTargets()
    {
        // No hemos visitado el target1
        if (!visited1) 
        {
            // Nos movemos hacia el target1
            transform.position = Vector3.MoveTowards(transform.position, targetObject1.transform.position, 0.01f);
            // Comprobamos si hemos llegado al target1, si es asi actualizamos el estado
            if (Vector3.Distance(transform.position, targetObject1.transform.position) < 0.01f)
            {
                visited1 = true;
            }
        }
        // Si hemos visitado el target1 nos movemos al target2
        else
        {
            // Nos movemos al target2
            transform.position = Vector3.MoveTowards(transform.position, targetObject2.transform.position, 0.01f);
            // Comprobamos si estamos en el target2, si es asi actualizamos el estado
            if (Vector3.Distance(transform.position, targetObject2.transform.position) < 0.01f)
            {
                visited1 = false;
            }
        }
    }
}
