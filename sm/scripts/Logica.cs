using System.Collections.Generic;
using UnityEngine;

public class Logica : MonoBehaviour
{
    //public static readonly Vector3 vector3 = new Vector3(-5, 1, -5);
    public List<Vector3> SpawnPointsPoli = new List<Vector3>{
        new Vector3(5,1,5),
        new Vector3(5,1,-5),
        new Vector3(-5,1,5)
    };
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
