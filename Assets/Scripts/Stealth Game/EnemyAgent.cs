using UnityEngine;

namespace Stealth_Game
{
    public class EnemyAgent : MonoBehaviour, IResettable
    {
        // patrol behaviour variables
        [SerializeField] private PatrolPath patrolPath;
        [SerializeField] private float waypointDwellTime = 3f;
        [SerializeField] private float waypointTolerance = 1f;
        [Range(0, 1)] [SerializeField] private float patrolSpeedFraction = 0.5f;
        [SerializeField] private float agentMaxSpeed = 6f;
        [SerializeField] private float rotationSpeed = 4f;

        // line of sight variables
        [SerializeField] private float viewRadius;
        [Range(0, 360)] [SerializeField] private float viewAngle;
        [SerializeField] private LayerMask obstacleMask;

        [SerializeField] private float meshResolution;
        [SerializeField] private MeshFilter viewMeshFilter;
        private Mesh _viewMesh;
        private float _viewStepAngleSize;
        private Vector3[] _vertices;
        private int[] _triangles;

        private bool _isDead;

        // cached variables
        private int _currentWaypointIndex;
        private float _timeSinceArriveAtWaypoints = Mathf.Infinity;
        private float _sqrViewRadius;
        private Vector3 _initPosition;
        private Quaternion _initRotation;
        private Renderer _renderer;
        private Collider _collider;

        public Vector3[] ViewPoints { get; private set; }
        public float ViewRadius => viewRadius;
        public float ViewAngle => viewAngle;

        private void Awake()
        {
            _sqrViewRadius = viewRadius * viewRadius;

            int stepCount = Mathf.RoundToInt(viewAngle * meshResolution);
            _viewStepAngleSize = viewAngle / stepCount;
            ViewPoints = new Vector3[stepCount + 1];

            int vertexCount = ViewPoints.Length + 1;
            _vertices = new Vector3[vertexCount];
            _triangles = new int[(vertexCount - 2) * 3];

            _viewMesh = new Mesh { name = "View Mesh" };
            viewMeshFilter.mesh = _viewMesh;

            var transform1 = transform;
            _initPosition = transform1.position;
            _initRotation = transform1.rotation;

            _renderer = GetComponent<Renderer>();
            _collider = GetComponent<Collider>();
        }

        public void UpdateEnemy()
        {
            if (_isDead) return;

            PatrolBehaviour();
            DrawFieldOfView();
            _timeSinceArriveAtWaypoints += Time.fixedDeltaTime;
        }

        public void KillAgent()
        {
            if (_isDead) return;

            _isDead = true;
            transform.position = Vector3.up;
            for (int i = 0; i < ViewPoints.Length; i++)
            {
                ViewPoints[i] = Vector3.up;
            }

            _viewMesh.Clear();

            _renderer.enabled = false;
            _collider.enabled = false;

            foreach (Transform child in transform)
            {
                child.gameObject.SetActive(false);
            }
        }

        public bool CanSeeTarget(Vector3 targetPosition)
        {
            if (_isDead) return false;

            var dirToPlayer = targetPosition - transform.position;

            if (dirToPlayer.sqrMagnitude > _sqrViewRadius) return false;

            if (Vector3.Angle(transform.forward, dirToPlayer) >= viewAngle / 2) return false;

            return !Physics.Raycast(transform.position, dirToPlayer, dirToPlayer.magnitude, obstacleMask);
        }

        private void PatrolBehaviour()
        {
            if (patrolPath is null) return;

            var agentPosition = transform.position;

            if (Vector3.Distance(agentPosition, CurrentWaypoint) < waypointTolerance)
            {
                _timeSinceArriveAtWaypoints = 0;
                _currentWaypointIndex = patrolPath.GetNextIndex(_currentWaypointIndex);
            }

            if (_timeSinceArriveAtWaypoints <= waypointDwellTime) return;

            agentPosition = Vector3.MoveTowards(agentPosition, CurrentWaypoint,
                agentMaxSpeed * patrolSpeedFraction * Time.deltaTime);
            transform.position = agentPosition;

            var direction = CurrentWaypoint - agentPosition;
            if (direction == Vector3.zero) return;

            var lookRotation = Quaternion.LookRotation(direction);
            transform.rotation =
                Quaternion.Slerp(transform.rotation, lookRotation, Time.deltaTime * rotationSpeed);
        }

        private Vector3 CurrentWaypoint => patrolPath.GetWaypoint(_currentWaypointIndex);

        private void DrawFieldOfView()
        {
            int lenght = ViewPoints.Length;
            _vertices[0] = Vector3.zero;
            for (int i = 0; i < lenght; i++)
            {
                float angle = transform.eulerAngles.y - viewAngle / 2 + _viewStepAngleSize * i;
                var viewPoint = ViewCast(angle);
                ViewPoints[i] = viewPoint;

                _vertices[i + 1] = transform.InverseTransformPoint(viewPoint);

                if (i >= lenght - 1) continue;

                int index = i * 3;
                _triangles[index] = 0;
                _triangles[index + 1] = i + 1;
                _triangles[index + 2] = i + 2;
            }

            _viewMesh.Clear();
            _viewMesh.vertices = _vertices;
            _viewMesh.triangles = _triangles;
            _viewMesh.RecalculateNormals();
        }

        private Vector3 ViewCast(float globalAngle)
        {
            var direction = DirFromAngle(globalAngle, true);
            return Physics.Raycast(transform.position, direction, out var hit, viewRadius, obstacleMask)
                ? hit.point
                : transform.position + direction * viewRadius;
        }

        private Vector3 DirFromAngle(float angleInDegrees, bool angleIsGlobal)
        {
            if (!angleIsGlobal)
                angleInDegrees += transform.eulerAngles.y;

            var angleRadians = angleInDegrees * Mathf.Deg2Rad;
            return new Vector3(Mathf.Sin(angleRadians), 0.0f, Mathf.Cos(angleRadians));
        }

        public void ResetAgent()
        {
            var transformRef = transform;
            transformRef.position = _initPosition;
            transformRef.rotation = _initRotation;

            DrawFieldOfView();
            _currentWaypointIndex = 0;
            _isDead = false;

            _renderer.enabled = true;
            _collider.enabled = true;

            foreach (Transform child in transformRef)
            {
                child.gameObject.SetActive(true);
            }
        }
    }
}

/*
 private struct ViewCastInfo
        {
            public bool Hit;
            public Vector3 EndPoint;
            public float Distance;
            public float Angle;

            public ViewCastInfo(bool hit, Vector3 endPoint, float distance, float angle)
            {
                Hit = hit;
                EndPoint = endPoint;
                Distance = distance;
                Angle = angle;
            }
        }

        private Vector3 DirFromAngle(float angleInDegrees, bool angleIsGlobal)
        {
            if (!angleIsGlobal)
                angleInDegrees += transform.eulerAngles.y;

            return new Vector3(Mathf.Sin(angleInDegrees * Mathf.Deg2Rad), 0, Mathf.Cos(angleInDegrees * Mathf.Deg2Rad));
        }

        private ViewCastInfo ViewCast(float globalAngle)
        {
            var direction = DirFromAngle(globalAngle, true);
            return Physics.Raycast(transform.position, direction, out var hit, viewRadius, obstacleMask)
                ? new ViewCastInfo(true, hit.point, hit.distance, globalAngle)
                : new ViewCastInfo(false, transform.position + direction * viewRadius, viewRadius, globalAngle);
        }

        private void DrawFieldOfView()
        {
            int stepCount = Mathf.RoundToInt(viewAngle * meshResolution);
            float stepAngleSize = viewAngle / stepCount;

            List<Vector3> viewPoints = new List<Vector3>(stepCount);
            ViewCastInfo oldViewCast = new ViewCastInfo();
            for (int i = 0; i < stepCount; i++)
            {
                float angle = transform.eulerAngles.y - viewAngle / 2 + stepAngleSize * i;
                ViewCastInfo newViewCast = ViewCast(angle);

                if (i > 0)
                {
                    bool edgeDistThresholdExceeded =
                        Mathf.Abs(oldViewCast.Distance - newViewCast.Distance) > edgeDistanceThreshold;
                    if (oldViewCast.Hit != newViewCast.Hit || (oldViewCast.Hit && edgeDistThresholdExceeded))
                    {
                        EdgeInfo edge = FindEdge(oldViewCast, newViewCast);
                        if (edge.pointA != Vector3.zero)
                        {
                            viewPoints.Add(edge.pointA);
                        }

                        if (edge.pointA != Vector3.zero)
                        {
                            viewPoints.Add(edge.pointB);
                        }
                    }
                }

                viewPoints.Add(newViewCast.EndPoint);
                oldViewCast = newViewCast;
            }

            int vertexCount = viewPoints.Count + 1;
            Vector3[] vertices = new Vector3[vertexCount];
            int[] triangles = new int[(vertexCount - 2) * 3];

            vertices[0] = Vector3.zero;
            for (int i = 0; i < vertexCount - 1; i++)
            {
                vertices[i + 1] = transform.InverseTransformPoint(viewPoints[i]);

                if (i < vertexCount - 2)
                {
                    triangles[i * 3] = 0;
                    triangles[i * 3 + 1] = i + 1;
                    triangles[i * 3 + 2] = i + 2;
                }
            }

            _viewMesh.Clear();
            _viewMesh.vertices = vertices;
            _viewMesh.triangles = triangles;
            _viewMesh.RecalculateNormals();
        }

        public int edgeResolveIterations;
        public float edgeDistanceThreshold;

        private EdgeInfo FindEdge(ViewCastInfo minViewCast, ViewCastInfo maxViewCast)
        {
            float minAngle = minViewCast.Angle;
            float maxAngle = maxViewCast.Angle;
            Vector3 minPoint = Vector3.zero;
            Vector3 maxPoint = Vector3.zero;

            for (int i = 0; i < edgeResolveIterations; i++)
            {
                float angle = (minAngle + maxAngle) / 2;
                ViewCastInfo newViewCast = ViewCast(angle);
                bool edgeDistThresholdExceeded =
                    Mathf.Abs(minViewCast.Distance - newViewCast.Distance) > edgeDistanceThreshold;
                if (newViewCast.Hit == minViewCast.Hit && !edgeDistThresholdExceeded)
                {
                    minAngle = angle;
                    minPoint = newViewCast.EndPoint;
                }
                else
                {
                    maxAngle = angle;
                    maxPoint = newViewCast.EndPoint;
                }
            }

            return new EdgeInfo(minPoint, maxPoint);
        }

        public struct EdgeInfo
        {
            public Vector3 pointA;
            public Vector3 pointB;


            public EdgeInfo(Vector3 pointA, Vector3 pointB)
            {
                this.pointA = pointA;
                this.pointB = pointB;
            }
        }
 */