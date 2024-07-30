using System.Collections.Generic;
using Stealth_Game;
using UnityEngine;

namespace Gym
{
    public class StealthGameEnv : Environment<Vector3>
    {
        [SerializeField] private Transform[] stealthLevels;
        [SerializeField] protected float passiveReward;
        [SerializeField] protected float goalReachedReward;
        [SerializeField] protected float spottedReward;
        [SerializeField] protected float assassinateReward;

        protected PlayerAgent _player;
        protected List<EnemyAgent> _enemies;
        protected Transform _goalTransform;

        private Dictionary<StealthLevels, Transform> _levelsTable;

        protected bool _envStarted;

        //cashed variables
        protected int _playerViewPoints;
        protected int _enemyCount;
        protected int _enemyViewPoints;

        protected float[] _resetObservation;

        public enum StealthLevels
        {
            LevelOne
        }

        public void LoadEnv(string enumName)
        {
        }

        protected override void Awake()
        {
            base.Awake();
            ActionLookup = new Vector3[]
            {
                Vector3.zero, Vector3.forward, Vector3.back, Vector3.right, Vector3.left,
                Vector3.forward + Vector3.right, Vector3.forward + Vector3.left, Vector3.back + Vector3.right,
                Vector3.back + Vector3.left, Vector3.up
            };

            for (int i = 0; i < ActionLookup.Length; i++)
            {
                ActionLookup[i].Normalize();
            }

            _enemies = new List<EnemyAgent>();
            foreach (var envTransform in AllEnvTransforms)
            {
                if (envTransform.CompareTag("Goal"))
                {
                    _goalTransform = envTransform;
                    continue;
                }

                var enemy = envTransform.GetComponent<EnemyAgent>();
                if (enemy)
                {
                    _enemies.Add(enemy);
                    continue;
                }

                if (_player) continue;
                var player = envTransform.GetComponent<PlayerAgent>();
                if (player)
                {
                    _player = player;
                }
            }

            _enemyCount = _enemies.Count;
        }

        protected virtual void Start()
        {
            if (_envStarted) return;

            _envStarted = true;

            _playerViewPoints = _player.ViewPoints.Length;
            if (_enemies.Count > 0)
            {
                _enemyViewPoints = _enemies[0].ViewPoints.Length;
            }

            ObservationLenght = 4;
            if (_player.ViewPoints != null)
            {
                //ObservationLenght += _player.ViewPoints.Length;
                ObservationLenght += _player.ViewPoints.Length * 2;
            }

            if (_enemies.Count <= 0 || _enemies[0].ViewPoints == null) return;

            ObservationLenght += _enemies.Count * 2;
            ObservationLenght += _enemies[0].ViewPoints.Length * 2 * _enemies.Count;
        }

        public override StepInfo Step(int actionIndex, bool skippFrame = false)
        {
            var action = ActionLookup[actionIndex];
            var observation = new float[ObservationLenght];
            var stepInfo = new StepInfo(observation, passiveReward, EpisodeLengthIndex > episodeLength);

            if (action.y != 0)
            {
                action.y = 0;
                if (_player.IterableObjects.Count > 0)
                {
                    var enemyToRemove = _player.IterableObjects[0].GetComponent<EnemyAgent>();
                    if (enemyToRemove)
                    {
                        enemyToRemove.KillAgent();
                        _player.IterableObjects.RemoveAt(0);
                        stepInfo.Reward = assassinateReward;
                    }
                }
            }

            var goalPosition = _goalTransform.position;
            observation[0] = NormalizePosition(goalPosition.x, true);
            observation[1] = NormalizePosition(goalPosition.z, false);

            _player.MovePlayer(action);
            _player.CheckObstacles();

            var playerPosition = _player.transform.position;
            observation[2] = NormalizePosition(playerPosition.x, true);
            observation[3] = NormalizePosition(playerPosition.z, false);

            int obsIndex = 4;
            for (int i = 0; i < _playerViewPoints; i++)
            {
                // observation[obsIndex] = NormalizeDistance(_player.ViewDistances[i]);
                // obsIndex++;
                var viewPoint = _player.ViewPoints[i];
                observation[obsIndex] = NormalizePosition(viewPoint.x, true);
                observation[obsIndex + 1] = NormalizePosition(viewPoint.z, false);
                obsIndex += 2;
            }

            for (var i = 0; i < _enemyCount; i++)
            {
                var enemy = _enemies[i];
                enemy.UpdateEnemy();
                var enemyPosition = enemy.transform.position;

                observation[obsIndex] = NormalizePosition(enemyPosition.x, true);
                observation[obsIndex + 1] = NormalizePosition(enemyPosition.z, false);
                obsIndex += 2;

                for (int j = 0; j < _enemyViewPoints; j++)
                {
                    var viewPoint = enemy.ViewPoints[j];
                    observation[obsIndex] = NormalizePosition(viewPoint.x, true);
                    observation[obsIndex + 1] = NormalizePosition(viewPoint.z, false);
                    obsIndex += 2;
                }

                if (!enemy.CanSeeTarget(playerPosition)) continue;

                stepInfo.Done = true;
                stepInfo.Reward = spottedReward;
            }

            if (_player.GoalReached)
            {
                stepInfo.Done = true;
                stepInfo.Reward = goalReachedReward;
            }

            EpisodeLengthIndex++;
            return stepInfo;
        }

        public override float[] ResetEnv()
        {
            if (!_envStarted)
            {
                Start();
            }
            else
            {
               BaseResetEnv();
            }
            
            Physics.SyncTransforms();

            _resetObservation = new float[ObservationLenght];

            var goalPosition = _goalTransform.position;
            _resetObservation[0] = NormalizePosition(goalPosition.x, true);
            _resetObservation[1] = NormalizePosition(goalPosition.z, false);

            var playerPosition = _player.transform.position;
            _resetObservation[2] = NormalizePosition(playerPosition.x, true);
            _resetObservation[3] = NormalizePosition(playerPosition.z, false);

            _player.CheckObstacles();
            int obsIndex = 4;

            for (int i = 0; i < _playerViewPoints; i++)
            {
                // _resetObservation[obsIndex] = NormalizeDistance(_player.ViewDistances[i]);
                // obsIndex++;
                var viewPoint = _player.ViewPoints[i];
                _resetObservation[obsIndex] = NormalizePosition(viewPoint.x, true);
                _resetObservation[obsIndex + 1] = NormalizePosition(viewPoint.z, false);
                obsIndex += 2;
            }

            for (var i = 0; i < _enemyCount; i++)
            {
                var enemy = _enemies[i];
                //TODO: There might not be a need to call update enemy here. Test for other environments as well
                enemy.UpdateEnemy();
                var enemyPosition = enemy.transform.position;

                _resetObservation[obsIndex] = NormalizePosition(enemyPosition.x, true);
                _resetObservation[obsIndex + 1] = NormalizePosition(enemyPosition.z, false);
                obsIndex += 2;

                for (int j = 0; j < _enemyViewPoints; j++)
                {
                    var viewPoint = enemy.ViewPoints[j];
                    _resetObservation[obsIndex] = NormalizePosition(viewPoint.x, true);
                    _resetObservation[obsIndex + 1] = NormalizePosition(viewPoint.z, false);
                    obsIndex += 2;
                }
            }

            return _resetObservation;
        }
    }
}
