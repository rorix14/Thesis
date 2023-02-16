using UnityEngine;

namespace Stealth_Game
{
    public class PatrolPath : MonoBehaviour
    {
        private const float WaypointGizmosRadius = 0.3f;
        private Vector3[] _waypointPositions;

        private void Awake()
        {
            _waypointPositions = new Vector3[transform.childCount];
            for (int i = 0; i < transform.childCount; i++)
            {
                _waypointPositions[i] = transform.GetChild(i).position;
            }
        }

        public int GetNextIndex(int i) => i + 1 < _waypointPositions.Length ? i + 1 : 0;

        public Vector3 GetWaypoint(int i) => _waypointPositions[i];

        private void OnDrawGizmos()
        {
            for (int i = 0; i < transform.childCount; i++)
            {
                Gizmos.DrawSphere(transform.GetChild(i).position, WaypointGizmosRadius);
                Gizmos.DrawLine(transform.GetChild(i).position,
                    transform.GetChild((i + 1) % transform.childCount).position);
            }
        }
    }
}