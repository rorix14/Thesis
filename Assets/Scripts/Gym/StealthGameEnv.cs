using System.Collections.Generic;
using Stealth_Game;
using UnityEngine;

namespace Gym
{
    public class StealthGameEnv : Environment<Vector3>
    {
        [SerializeField] private Transform[] stealthLevels;
        [SerializeField] private float passiveReward;
        [SerializeField] private float goalReachedReward;
        [SerializeField] private float spottedReward;

        private PlayerAgent _player;
        private List<EnemyAgent> _enemies;
        private Vector3 _goalPosition;

        private Dictionary<StealthLevels, Transform> _levelsTable;

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
            _enemies = new List<EnemyAgent>();

            foreach (var envTransform in AllEnvTransforms)
            {
                if (envTransform.CompareTag("Goal"))
                {
                    _goalPosition = envTransform.position;
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

            CurrentObservation = new float[4 + _enemies.Count * 2];
            CurrentObservation[0] = _goalPosition.x;
            CurrentObservation[1] = _goalPosition.z;
        }

        public override StepInfo Step(Vector3 action)
        {
            _player.MovePlayer(action);
            var playerPosition = _player.transform.position;
            CurrentObservation[2] = playerPosition.x;
            CurrentObservation[3] = playerPosition.z;
                
            var stepInfo = new StepInfo(CurrentObservation, passiveReward, EpisodeLengthIndex > episodeLength);

            if (_player.GoalReached)
            {
                stepInfo.Done = true;
                stepInfo.Reward = goalReachedReward;
                return stepInfo;
            }

            for (var i = 0; i < _enemies.Count; i++)
            {
                var enemy = _enemies[i];
                enemy.UpdateEnemy();
                var enemyPosition = enemy.transform.position;
                int obsIndex = i * 2 + 4;
                CurrentObservation[obsIndex] = enemyPosition.x;
                CurrentObservation[obsIndex + 1] = enemyPosition.z;
                
                if (!enemy.CanSeeTarget(playerPosition)) continue;

                stepInfo.Done = true;
                stepInfo.Reward = spottedReward;
                return stepInfo;
            }

            EpisodeLengthIndex++;
            return stepInfo;
        }

        public override float[] ResetEnv()
        {
           base.ResetEnv();

            var playerPosition = _player.transform.position;
            CurrentObservation[2] = playerPosition.x;
            CurrentObservation[3] = playerPosition.z;
            
            for (var i = 0; i < _enemies.Count; i++)
            {
                var enemyPosition = _enemies[i].transform.position;

                int obsIndex = i * 2 + 4;
                CurrentObservation[obsIndex] = enemyPosition.x;
                CurrentObservation[obsIndex + 1] = enemyPosition.z;
            }

            return CurrentObservation;
        }
    }
}