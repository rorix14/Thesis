using Stealth_Game;

namespace Gym
{
    public class DistributedStealthGameEnv : StealthGameEnv
    {
        private int _populationSize;
        private PlayerAgent[] _playerAgents;
        private EnemyAgent[] _agentAssassinated;

        //cashed variables
        private int _viewPointsYXLenght;
        private int _enemyObservationSize;

        private float[] _normalizedPlayerViewPoints;
        private float[] _enemyObservation;

        protected override void Start()
        {
            base.Start();

            _viewPointsYXLenght = _playerViewPoints * 2;
            _normalizedPlayerViewPoints = new float[_viewPointsYXLenght];

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
                var newPlayer = Instantiate(_player);
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
                
                
                if (action.y != 0)
                {
                    action.y = 0;
                    if (currentPlayer.IterableObjects.Count > 0)
                    {
                        var enemyToRemove = _player.IterableObjects[0].GetComponent<EnemyAgent>();
                        var hasKilledEnemy = false;
                        for (int j = 0; j < _enemyCount; j++)
                        {
                            if (_agentAssassinated[i * _enemyCount + j] == enemyToRemove)
                            {
                                hasKilledEnemy = true;
                                break;
                            }
                        }

                        if (!hasKilledEnemy)
                        {
                            _agentAssassinated[i] = enemyToRemove;
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
                    var viewPoint = _player.ViewPoints[i];
                    currentStep.Observations[i, obsIndex] = NormalizePosition(viewPoint.x, true);
                    currentStep.Observations[i, obsIndex + 1] = NormalizePosition(viewPoint.z, false);
                    obsIndex += 2;
                }

                for (int j = 0; j < _enemyCount; j++)
                {
                    var enemy = _enemies[i];
                    if (_agentAssassinated[i * _enemyCount + j] == enemy) continue;

                    index = j * (_enemyCount + _enemyViewPoints);
                    currentStep.Observations[i, obsIndex + index] = _enemyObservation[j * (_enemyCount + _enemyViewPoints)];

                    if (!enemy.CanSeeTarget(playerPosition)) continue;

                    currentStep.Dones[i] = true;
                    currentStep.Rewards[i] = spottedReward;
                }

                if (!currentPlayer.GoalReached) continue;

                currentStep.Dones[i] = true;
                currentStep.Rewards[i] = goalReachedReward;
            }

            EpisodeLengthIndex++;

            return new DistributedStepInfo();
        }

        public override float[,] DistributedResetEnv()
        {
            if (!_envStarted)
            {
                Start();
            }
            else
            {
                base.DistributedResetEnv();
            }

            //TODO: Should just return a normal array
            var observationBatch = new float[_populationSize, ObservationLenght];

            //TODO: can just put this values into an array, and fill the observationBatch variable later in one loop
            var goalPosition = _goalTransform.position;
            var goalPositionX = NormalizePosition(goalPosition.x, true);
            var goalPositionZ = NormalizePosition(goalPosition.z, false);

            var playerPosition = _player.transform.position;
            var playerPositionX = NormalizePosition(playerPosition.x, true);
            var playerPositionZ = NormalizePosition(playerPosition.z, false);

            _player.CheckObstacles();
            int index = 0;
            for (int i = 0; i < _playerViewPoints; i++)
            {
                var viewPoint = _player.ViewPoints[i];
                index = i * 2;
                _normalizedPlayerViewPoints[index] = NormalizePosition(viewPoint.x, true);
                _normalizedPlayerViewPoints[index + 1] = NormalizePosition(viewPoint.z, true);
            }

            index = 0;
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

            //TODO: no need to reset done individuals
            for (int i = 0; i < _populationSize; i++)
            {
                _playerAgents[i].SetPosition(playerPosition);

                for (int j = 0; j < _enemyCount; j++)
                {
                    _agentAssassinated[i * _enemyCount + j] = null;
                }

                observationBatch[i, 0] = goalPositionX;
                observationBatch[i, 1] = goalPositionZ;

                observationBatch[i, 2] = playerPositionX;
                observationBatch[i, 3] = playerPositionZ;

                int obsIndex = 4;
                for (int j = 0; j < _viewPointsYXLenght; j++)
                {
                    observationBatch[i, obsIndex] = _normalizedPlayerViewPoints[j];
                    obsIndex++;
                }

                for (int j = 0; j < _enemyObservationSize; j++)
                {
                    observationBatch[i, obsIndex] = _enemyObservation[j];
                    obsIndex++;
                }
            }

            return observationBatch;
        }
    }
}