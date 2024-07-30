using Stealth_Game;
using UnityEngine;

namespace Gym
{
    public class DistributedStealthGameEnv : StealthGameEnv
    {
        private int _populationSize;
        private PlayerAgent[] _playerAgents;
        private EnemyAgent[] _agentAssassinated;

        //cashed variables
        private int _enemyObservationSize;
        private int _singleEnemyObservationSize;

        private float[] _enemyObservation;

        private float[,] _resetObservationBatch;

        protected override void Start()
        {
            base.Start();

            _singleEnemyObservationSize = 2 + _enemyViewPoints * 2;
            _enemyObservationSize = _enemyCount * 2 + _enemyCount * _enemyViewPoints * 2;
            _enemyObservation = new float[_enemyObservationSize];
        }

        public void CreatePopulation(int populationSize)
        {
            if (!_player || populationSize == 0) return;

            _populationSize = populationSize;

            _agentAssassinated = new EnemyAgent[populationSize * _enemyCount];

            _playerAgents = new PlayerAgent[populationSize];
            _playerAgents[0] = _player;
            for (int i = 1; i < populationSize; i++)
            {
                var newPlayer = Instantiate(_player, _player.transform.parent);
                _playerAgents[i] = newPlayer;
                _resettables.Add(newPlayer);
            }
        }

        //TODO: Test with jobs on player movement and player raycast
        public override DistributedStepInfo DistributedStep(int[] actions)
        {
            var currentStep = new DistributedStepInfo(new float[_populationSize, ObservationLenght],
                new float[_populationSize], new bool[_populationSize]);

            var goalPosition = _goalTransform.position;
            var goalPositionX = NormalizePosition(goalPosition.x, true);
            var goalPositionZ = NormalizePosition(goalPosition.z, false);

            var index = 0;
            for (int i = 0; i < _enemyCount; i++)
            {
                var enemy = _enemies[i];
                enemy.UpdateEnemy();
                var enemyPosition = enemy.transform.position;

                _enemyObservation[index] = NormalizePosition(enemyPosition.x, true);
                _enemyObservation[index + 1] = NormalizePosition(enemyPosition.z, false);
                index += 2;

                for (int j = 0; j < _enemyViewPoints; j++)
                {
                    var viewPoint = enemy.ViewPoints[j];
                    _enemyObservation[index] = NormalizePosition(viewPoint.x, true);
                    _enemyObservation[index + 1] = NormalizePosition(viewPoint.z, false);
                    index += 2;
                }
            }

            for (int i = 0; i < _populationSize; i++)
            {
                var actionIndex = actions[i];

                // TODO: should do something more stable like: index being out of bounds
                if (actionIndex == -1)
                {
                    currentStep.Rewards[i] = 0;
                    currentStep.Dones[i] = true;
                    continue;
                }

                var action = ActionLookup[actionIndex];
                var currentPlayer = _playerAgents[i];

                currentStep.Rewards[i] = passiveReward;
                currentStep.Dones[i] = EpisodeLengthIndex > episodeLength;

                var assassinatedIndex = i * _enemyCount;
                if (action.y != 0)
                {
                    action.y = 0;
                    if (currentPlayer.IterableObjects.Count > 0)
                    {
                        var enemyToRemove = currentPlayer.IterableObjects[0].GetComponent<EnemyAgent>();
                        var hasKilledEnemy = false;
                        for (int j = 0; j < _enemyCount; j++)
                        {
                            if (_agentAssassinated[assassinatedIndex + j] != enemyToRemove) continue;

                            hasKilledEnemy = true;
                            break;
                        }

                        if (!hasKilledEnemy)
                        {
                            _agentAssassinated[assassinatedIndex] = enemyToRemove;
                            currentPlayer.IterableObjects.RemoveAt(0);
                            currentStep.Rewards[i] = assassinateReward;
                        }
                    }
                }

                currentStep.Observations[i, 0] = goalPositionX;
                currentStep.Observations[i, 1] = goalPositionZ;

                currentPlayer.MovePlayer(action);
                currentPlayer.CheckObstacles();

                var playerPosition = currentPlayer.transform.position;
                currentStep.Observations[i, 2] = NormalizePosition(playerPosition.x, true);
                currentStep.Observations[i, 3] = NormalizePosition(playerPosition.z, false);

                int obsIndex = 4;
                for (int j = 0; j < _playerViewPoints; j++)
                {
                    var viewPoint = currentPlayer.ViewPoints[j];
                    currentStep.Observations[i, obsIndex] = NormalizePosition(viewPoint.x, true);
                    currentStep.Observations[i, obsIndex + 1] = NormalizePosition(viewPoint.z, false);
                    obsIndex += 2;
                }

                for (int j = 0; j < _enemyCount; j++)
                {
                    var enemy = _enemies[j];
                    if (_agentAssassinated[assassinatedIndex + j] == enemy) continue;

                    index = j * _singleEnemyObservationSize;
                    for (int k = 0; k < _singleEnemyObservationSize; k++)
                    {
                        currentStep.Observations[i, obsIndex + k + index] = _enemyObservation[k + index];
                    }

                    if (!enemy.CanSeeTarget(playerPosition)) continue;

                    currentStep.Dones[i] = true;
                    currentStep.Rewards[i] = spottedReward;
                }

                if (!currentPlayer.GoalReached) continue;

                currentStep.Dones[i] = true;
                currentStep.Rewards[i] = goalReachedReward;
            }

            EpisodeLengthIndex++;

            return currentStep;
        }

        public override float[,] DistributedResetEnv()
        {
            if (!_envStarted)
            {
                Start();
            }
            else
            {
                BaseResetEnv();
            }

            //TODO: Should just return a normal array
            _resetObservationBatch = new float[_populationSize, ObservationLenght];
            
            //TODO: some arrays might only need to be initialized once, must check once all distributed algorithms have been implemented
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
                var viewPoint = _player.ViewPoints[i];
                _resetObservation[obsIndex] = NormalizePosition(viewPoint.x, true);
                _resetObservation[obsIndex + 1] = NormalizePosition(viewPoint.z, true);
                obsIndex += 2;
            }

            for (int i = 0; i < _enemyCount; i++)
            {
                var enemy = _enemies[i];
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

            for (int i = 0; i < _populationSize; i++)
            {
                _playerAgents[i].SetPosition(playerPosition);

                for (int j = 0; j < ObservationLenght; j++)
                {
                    _resetObservationBatch[i, j] = _resetObservation[j];
                }

                for (int j = 0; j < _enemyCount; j++)
                {
                    _agentAssassinated[i * _enemyCount + j] = null;
                }
            }

            Physics.SyncTransforms();

            return _resetObservationBatch;
        }
    }
}