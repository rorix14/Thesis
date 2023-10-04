using System;
using System.Collections.Generic;
using NN;
using UnityEngine;

namespace Algorithms.NE
{
    public class ES
    {
        protected readonly NetworkModel _networkModel;
        protected readonly int _numberOfActions;
        protected readonly int _batchSize;

        protected readonly int[] _sampledActions;

        protected readonly float[] _episodeRewards;
        protected readonly bool[] _completedAgents;

        //Cashed variables
        protected float[,] _modelPredictions;

        protected readonly float[,] _episodeRewardUpdate;
        protected int _finishedIndividuals;
        protected float _episodeRewardMean;
        protected float _episodeBestReward;

        private readonly int[] _rewardsIndexKeys;

        protected readonly float[] _adjustedPopulationFitness;

        //Novelty Search
        private readonly List<Vector2> _archive;
        private float _noveltyThreshold;
        private readonly float _noveltyThresholdMinValue;
        private int _timeout;
        private readonly int _neighboursToCheck;
        private readonly float[] _noveltyScores;
        private readonly List<float>[] _agentsArchiveDistances;
        protected readonly float _noveltyRelevance;
        protected readonly bool _useNovelty;

        public float EpisodeRewardMean => _episodeRewardMean;
        public float EpisodeBestReward => _episodeBestReward;
        public float FinishedIndividuals => _finishedIndividuals;

        public ES(NetworkModel networkModel, int numberOfActions, int batchSize, float noveltyRelevance = 0)
        {
            _networkModel = networkModel;
            _numberOfActions = numberOfActions;
            _batchSize = batchSize;

            _sampledActions = new int[batchSize];

            _episodeRewards = new float[batchSize];
            _completedAgents = new bool[batchSize];

            _episodeRewardUpdate = new float[1, batchSize];

            _rewardsIndexKeys = new int[batchSize];

            _adjustedPopulationFitness = new float[batchSize];

            //NS 
            _archive = new List<Vector2>(batchSize);
            _noveltyRelevance = noveltyRelevance;
            // one fourth of the max possible distance
            // _noveltyThreshold = 98f;
            // _noveltyThresholdMinValue = 20f; 
            _noveltyThreshold = 5f;
            _noveltyThresholdMinValue = 1f;
            _timeout = 0;
            _neighboursToCheck = 10;
            _noveltyScores = new float[batchSize];
            _agentsArchiveDistances = new List<float>[batchSize];
            _useNovelty = _noveltyRelevance > 0;
        }

        public virtual int[] SamplePopulationActions(float[,] states)
        {
            _modelPredictions = _networkModel.Predict(states);

            for (int i = 0; i < _batchSize; i++)
            {
                if (_completedAgents[i])
                {
                    _sampledActions[i] = -1;
                    continue;
                }

                var maxAction = _modelPredictions[i, 0];
                var maxIndex = 0;
                for (int j = 1; j < _numberOfActions; j++)
                {
                    var actionValue = _modelPredictions[i, j];
                    if (maxAction > actionValue) continue;

                    maxAction = actionValue;
                    maxIndex = j;
                }

                _sampledActions[i] = maxIndex;
            }

            return _sampledActions;
        }

        public void AddExperience(float[] rewards, bool[] dones)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                if (_completedAgents[i]) continue;

                var done = dones[i];
                _episodeRewards[i] += rewards[i];
                _completedAgents[i] = done;
                _finishedIndividuals += done ? 1 : 0;
            }
        }

        public void SoftReset()
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _completedAgents[i] = false;
            }

            _finishedIndividuals = 0;
        }

        public virtual void Train()
        {
            _episodeRewardMean = 0f;
            _episodeBestReward = float.MinValue;
            for (int i = 0; i < _batchSize; i++)
            {
                var episodeReward = _episodeRewards[i];
                //_episodeRewardUpdate[0, i] = episodeReward;
                _episodeRewardUpdate[0, i] = _useNovelty ? _adjustedPopulationFitness[i] : episodeReward;
                _episodeRewardMean += episodeReward;
                _episodeBestReward = episodeReward > _episodeBestReward ? episodeReward : _episodeBestReward;
                _completedAgents[i] = false;
                _episodeRewards[i] = 0f;
            }

            //RankRewards();

            _episodeRewardMean /= _batchSize;
            _finishedIndividuals = 0;

            _networkModel.Update(_episodeRewardUpdate);
        }

        public void DoNoveltySearch(Vector2[] agentsFinalPositions)
        {
            //TODO: needs to work if we want to do multiple episodes before training, example Moving Goal scene
            var addedToArchive = 0;
            for (int i = 0; i < _batchSize; i++)
            {
                var agentPosition = agentsFinalPositions[i];
                var archiveSize = _archive.Count;
                _agentsArchiveDistances[i] = new List<float>(archiveSize + _batchSize);

                var minDistance = float.MaxValue;
                for (int j = 0; j < archiveSize; j++)
                {
                    var archivePos = _archive[j];
                    var xDist = agentPosition.x - archivePos.x;
                    var yDist = agentPosition.y - archivePos.y;
                    var currentDistance = (float)Math.Sqrt(xDist * xDist + yDist * yDist);
                    _agentsArchiveDistances[i].Add(currentDistance);

                    if (minDistance < currentDistance) continue;

                    minDistance = currentDistance;
                }

                if (minDistance > _noveltyThreshold)
                {
                    _archive.Add(agentPosition);
                    addedToArchive++;
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    var currentPos = agentsFinalPositions[j];
                    var xDist = agentPosition.x - currentPos.x;
                    var yDist = agentPosition.y - currentPos.y;
                    _agentsArchiveDistances[i].Add((float)Math.Sqrt(xDist * xDist + yDist * yDist));
                }

                _agentsArchiveDistances[i].Sort();

                var distancesSum = 0f;
                for (int j = 1; j < _neighboursToCheck; j++)
                {
                    distancesSum += _agentsArchiveDistances[i][j];
                }

                _noveltyScores[i] = (distancesSum + 1) / (_neighboursToCheck - 1);
            }

            NormalizeRewards();

            if (addedToArchive == 0)
            {
                _timeout++;
                if (_timeout <= 10) return;

                _timeout = 0;
                _noveltyThreshold *= 0.9f;
                if (_noveltyThreshold < _noveltyThresholdMinValue)
                {
                    _noveltyThreshold = _noveltyThresholdMinValue;
                }
            }
            else
            {
                _timeout = 0;
                if (addedToArchive > 4)
                {
                    _noveltyThreshold *= 1.2f;
                }
            }
        }

        private void NormalizeRewards()
        {
            var rewardMin = float.MaxValue;
            var rewardMax = float.MinValue;
            var noveltyMin = float.MaxValue;
            var noveltyMax = float.MinValue;
            for (int i = 0; i < _batchSize; i++)
            {
                var currentReward = _episodeRewards[i];
                if (rewardMin > currentReward)
                {
                    rewardMin = currentReward;
                }

                if (rewardMax < currentReward)
                {
                    rewardMax = currentReward;
                }

                var currentNovelty = _noveltyScores[i];
                if (noveltyMin > currentNovelty)
                {
                    noveltyMin = currentNovelty;
                }

                if (noveltyMax < currentNovelty)
                {
                    noveltyMax = currentNovelty;
                }
            }

            var rewardRange = rewardMax - rewardMin;
            var noveltyRange = noveltyMax - noveltyMin;

            var rewardIsDifferent = Math.Abs(rewardMax - rewardMin) > 0f;
            var noveltyIsDifferent = Math.Abs(noveltyMax - noveltyMin) > 0f;
            for (int i = 0; i < _batchSize; i++)
            {
                var normReward = rewardIsDifferent ? (_episodeRewards[i] - rewardMin) / rewardRange : 0.5f;
                var normNovelty = noveltyIsDifferent ? (_noveltyScores[i] - noveltyMin) / noveltyRange : 0.5f;
                _adjustedPopulationFitness[i] = (1 - _noveltyRelevance) * normReward + _noveltyRelevance * normNovelty;
            }
        }

        private void RankRewards()
        {
            for (int i = 0; i < _batchSize; i++) _rewardsIndexKeys[i] = i;

            Array.Sort(_episodeRewards, _rewardsIndexKeys);

            var sumRanks = 0;
            var dupCount = 0;

            for (int i = 0; i < _batchSize; i++)
            {
                sumRanks += i;
                dupCount += 1;

                var episodeReward = _episodeRewards[i];
                if (i == _batchSize - 1 || Math.Abs(episodeReward - _episodeRewards[i + 1]) > 0.0f)
                {
                    var averageRank = sumRanks / (float)dupCount + 1;
                    for (int j = i - dupCount + 1; j < i + 1; j++)
                    {
                        _episodeRewardUpdate[0, _rewardsIndexKeys[j]] = (averageRank - 1) / (_batchSize - 1) - 0.5f;
                    }

                    sumRanks = 0;
                    dupCount = 0;
                }

                _episodeRewards[i] = 0f;
            }
        }

        // public void ReduceNoise(float noiseStd)
        // {
        //     _esModel.SetNoiseStd(noiseStd);
        // }

        public void Dispose()
        {
            _networkModel?.Dispose();
        }
    }
}