using System;
using NN;

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

        public float EpisodeRewardMean => _episodeRewardMean;
        public float EpisodeBestReward => _episodeBestReward;
        public float FinishedIndividuals => _finishedIndividuals;

        public ES(NetworkModel networkModel, int numberOfActions, int batchSize)
        {
            _networkModel = networkModel;
            _numberOfActions = numberOfActions;
            _batchSize = batchSize;

            _sampledActions = new int[batchSize];

            _episodeRewards = new float[batchSize];
            _completedAgents = new bool[batchSize];

            _episodeRewardUpdate = new float[1, batchSize];

            _rewardsIndexKeys = new int[batchSize];
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
                _episodeRewardMean += episodeReward;
                _episodeBestReward = episodeReward > _episodeBestReward ? episodeReward : _episodeBestReward;
                _completedAgents[i] = false;
                //_episodeRewards[i] = 0f;
            }

            RankRewards();

            _episodeRewardMean /= _batchSize;
            _finishedIndividuals = 0;

            _networkModel.Update(_episodeRewardUpdate);
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