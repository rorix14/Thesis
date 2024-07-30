using Stealth_Game;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;

namespace Gym
{
    public class JobStealthGameEnv : StealthGameEnv
    {
        private int _populationSize;
        private PlayerAgent[] _playerAgents;
        private EnemyAgent[] _agentAssassinated;

        private MovePlayerJob _movePlayerJob;
        private TransformAccessArray _playersTransforms;
        private NativeArray<RaycastHit> _playersWallChecks;

        //cashed variables
        private int _enemyObservationSize;
        private int _singleEnemyObservationSize;

        private float[] _enemyObservation;

        private float[,] _resetObservationBatch;

        private Vector2[] _playerPositions;

        protected override void Start()
        {
            base.Start();

            _singleEnemyObservationSize = 2 + _enemyViewPoints * 2;
            _enemyObservationSize = _enemyCount * 2 + _enemyCount * _enemyViewPoints * 2;
            _enemyObservation = new float[_enemyObservationSize];
        }

        public void CreatePopulation(int populationSize)
        {
            //TODO: should have a better create population method, currently does not support algorithm tests with different
            //population sizes, a solution would be to check if the previous population size is the same as the new one,
            //if so there is no need to run this function, if not players and must be created or destroyed accordingly
            //and array values must be disposed 
            if (!_player || populationSize == 0 || _populationSize == populationSize) return;
            
            //TODO: second condition might not be necessary
            if (_populationSize > 0 && _populationSize != populationSize)
            {
                Dispose();
                for (int i = 0; i < _populationSize; i++)
                {
                    var currentPlayer = _playerAgents[i];
                    if (_player == currentPlayer) continue;
                    
                    _resettables.Remove(currentPlayer);
                    Destroy(currentPlayer.gameObject);
                }
            }

            _populationSize = populationSize;

            _playerPositions = new Vector2[populationSize];

            _agentAssassinated = new EnemyAgent[populationSize * _enemyCount];

            var tempPlayerTransforms = new Transform[populationSize];

            _playerAgents = new PlayerAgent[populationSize];
            _playerAgents[0] = _player;
            tempPlayerTransforms[0] = _player.transform;

            for (int i = 1; i < populationSize; i++)
            {
                var newPlayer = Instantiate(_player, _player.transform.parent);
                _playerAgents[i] = newPlayer;
                _resettables.Add(newPlayer);
                newPlayer.name += i;

                tempPlayerTransforms[i] = newPlayer.transform;
            }

            var viewPointsLength = _player.ViewPoints.Length;
            _movePlayerJob = new MovePlayerJob
            {
                ActionLookup = new NativeArray<Vector3>(ActionLookup, Allocator.Persistent),
                RaycastCommands =
                    new NativeArray<RaycastCommand>(populationSize * viewPointsLength, Allocator.Persistent),
                Actions = new NativeArray<int>(populationSize, Allocator.Persistent),
                PlayerVelocity = new NativeArray<Vector3>(populationSize, Allocator.Persistent),

                WallChecks = viewPointsLength,
                ViewStepSize = 360 / (float)_player.WallChecks,
                ViewRadius = _player.ViewRadius,
                ObstacleMask = _player.ObstacleMask,
                MoveSpeed = _player.MoveSpeed,
                RotationSpeed = _player.RotationSpeed,
                FixedDeltaTime = Time.fixedDeltaTime,
                DegreeToRad = Mathf.Deg2Rad
            };

            _playersTransforms = new TransformAccessArray(tempPlayerTransforms);
            _playersWallChecks = new NativeArray<RaycastHit>(populationSize * viewPointsLength, Allocator.Persistent);
        }

        public override DistributedStepInfo DistributedStep(int[] actions)
        {
            _movePlayerJob.Actions.CopyFrom(actions);

            // jobs
            var playerJob = _movePlayerJob.Schedule(_playersTransforms);
            var playersWallCheckJob =
                RaycastCommand.ScheduleBatch(_movePlayerJob.RaycastCommands, _playersWallChecks, 16, playerJob);

            // normal logic
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

            playersWallCheckJob.Complete();

            for (int i = 0; i < _populationSize; i++)
            {
                var actionIndex = actions[i];

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

                currentPlayer.SoftMovePlayer(_movePlayerJob.PlayerVelocity[i]);
                var playerPosition = currentPlayer.transform.position;
                currentStep.Observations[i, 2] = NormalizePosition(playerPosition.x, true);
                currentStep.Observations[i, 3] = NormalizePosition(playerPosition.z, false);

                var obsIndex = 4;
                var playerWallCheckIndex = i * _playerViewPoints;
                for (int j = 0; j < _playerViewPoints; j++)
                {
                    var viewPoint = _playersWallChecks[playerWallCheckIndex + j].point;
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

        [BurstCompile]
        private struct MovePlayerJob : IJobParallelForTransform
        {
            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<RaycastCommand> RaycastCommands;

            [ReadOnly] public NativeArray<int> Actions;
            [ReadOnly] public NativeArray<Vector3> ActionLookup;
            [WriteOnly] public NativeArray<Vector3> PlayerVelocity;

            public int WallChecks;
            public float ViewStepSize;
            public float ViewRadius;
            public LayerMask ObstacleMask;

            public float MoveSpeed;
            public float RotationSpeed;
            public float FixedDeltaTime;
            public float DegreeToRad;

            public void Execute(int index, TransformAccess transform)
            {
                var pos = transform.position;

                var action = Actions[index];
                var done = action == -1;

                var playerRaycastIndex = index * WallChecks;
                for (int i = 0; i < WallChecks; i++)
                {
                    if (done)
                    {
                        RaycastCommands[playerRaycastIndex + i] = new RaycastCommand();
                        continue;
                    }

                    var angleRadians = DegreeToRad * (ViewStepSize * i);
                    var direction = new Vector3(math.sin(angleRadians), 0f, math.cos(angleRadians));
                    RaycastCommands[playerRaycastIndex + i] =
                        new RaycastCommand(pos, direction, ViewRadius, ObstacleMask);
                }

                if (done) return;

                var movementDirection = ActionLookup[action];
                movementDirection.y = 0f;

                PlayerVelocity[index] = movementDirection * MoveSpeed * FixedDeltaTime;

                if (movementDirection == Vector3.zero) return;

                var lookRotation = Quaternion.LookRotation(movementDirection, Vector3.up);
                transform.rotation = Quaternion.Slerp(transform.rotation, lookRotation, FixedDeltaTime * RotationSpeed);
            }
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

        public Vector2[] GetPlayersPositions()
        {
            for (int i = 0; i < _populationSize; i++)
            {
                var position = _playersTransforms[i].position;
                _playerPositions[i] = new Vector2(position.x, position.z);
            }

            return _playerPositions;
        }

        private void Dispose()
        {
            if (_movePlayerJob.PlayerVelocity.Length == 0) return;

            _movePlayerJob.Actions.Dispose();
            _movePlayerJob.ActionLookup.Dispose();
            _movePlayerJob.RaycastCommands.Dispose();
            _movePlayerJob.PlayerVelocity.Dispose();

            _playersTransforms.Dispose();
            _playersWallChecks.Dispose();
        }

        private void OnDestroy()
        {
            Dispose();
        }
    }
}