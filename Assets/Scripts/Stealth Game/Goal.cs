using UnityEngine;

namespace Stealth_Game
{
    public class Goal : MonoBehaviour, IResettable
    {
        [SerializeField] private Vector3[] startingPositions;
        
        public void ResetAgent()
        {
            var index = Random.Range(0, startingPositions.Length);
            transform.position = startingPositions[index];
        }
    }
}
