using UnityEngine;

namespace Stealth_Game
{
    public class PlayerAgent : MonoBehaviour, IResettable
    {
        [SerializeField] private float moveSpeed = 5f;
        [SerializeField] private float rotationSpeed = 10f;

        //cached variables
        private CharacterController _characterController;
        private Vector3 _moveDirection;
        private Vector3 _initPosition;
        private Quaternion _initRotation;

        public bool GoalReached { get; private set; }

        private void Awake()
        {
            _characterController = GetComponent<CharacterController>();
            var transformRef = transform;
            _initPosition = transformRef.position;
            _initRotation = transformRef.rotation;
        }

        public void MovePlayer(Vector3 movementDir)
        {
            _moveDirection = movementDir;
            _moveDirection.Normalize();
            _characterController.Move(_moveDirection * (moveSpeed * Time.fixedDeltaTime));

            if (_moveDirection == Vector3.zero) return;

            var lookRotation = Quaternion.LookRotation(_moveDirection, Vector3.up);
            transform.rotation =
                Quaternion.Slerp(transform.rotation, lookRotation, Time.fixedDeltaTime * rotationSpeed);
        }

        private void OnTriggerExit(Collider other)
        {
            if (!other.CompareTag("Goal")) return;
            
            GoalReached = false;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (!other.CompareTag("Goal")) return;

            GoalReached = true;
        }

        public void ResetAgent()
        {
            _moveDirection = Vector3.zero;
            var transformRef = transform;
            transformRef.position = _initPosition;
            transformRef.rotation = _initRotation;
            Physics.SyncTransforms();
            GoalReached = false;
        }
    }
}