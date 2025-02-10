using System.Collections.Generic;
using UnityEngine;

public class scriptPoli : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start(List<Vector3> puntoSpawn)
    {
        Logica logica = GameObject.FindGameObjectWithTag("Logica").GetComponent<Logica>();
        List<Vector3> puntoSpawn =  logica.SpawnPointsPoli;

        Random rnd = new()    
        int index = Random.Next(puntoSpawn.Count);

        //Debug.Log("lista: " + puntoSpawn[index]);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
