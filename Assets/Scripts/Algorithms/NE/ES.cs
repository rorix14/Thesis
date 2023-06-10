using System;

namespace Algorithms.NE
{
    public class ES
    {
        private readonly ESModel _networkModel;
        private readonly int _numberOfActions;
        private readonly int _batchSize;

        private readonly int[] _sampledActions;

        private readonly float[] _episodeRewards;
        private readonly bool[] _completedAgents;

        //Cashed variables
        private float[,] _modelPredictions;

        private readonly float[,] _episodeRewardUpdate;
        private int _finishedIndividuals;
        private float _episodeRewardMean;

        private readonly int[] _rewardsIndexKeys;

        public float EpisodeRewardMean => _episodeRewardMean;

        //public float EpisodeRewardMean => _networkModel.RewardMean;
        public float FinishedIndividuals => _finishedIndividuals;

        public ES(ESModel networkModel, int numberOfActions, int batchSize)
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

        public int[] SamplePopulationActions(float[,] states)
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

        public void Train()
        {
            // for (int i = 0; i < _batchSize; i++)
            // {
            //     _episodeRewardUpdate[0, i] = _episodeRewards[i];
            //
            //     _episodeRewards[i] = 0f;
            //     _completedAgents[i] = false;
            // }

            RankRewards();

            _finishedIndividuals = 0;

            _networkModel.Update(_episodeRewardUpdate);
        }

        private void RankRewards()
        {
            for (int i = 0; i < _batchSize; i++) _rewardsIndexKeys[i] = i;

            Array.Sort(_episodeRewards, _rewardsIndexKeys);

            var sumRanks = 0;
            var dupCount = 0;

            _episodeRewardMean = 0f;

            for (int i = 0; i < _batchSize; i++)
            {
                sumRanks += i;
                dupCount += 1;
                if (i == _batchSize - 1 || Math.Abs(_episodeRewards[i] - _episodeRewards[i + 1]) > 0.0f)
                {
                    var averageRank = sumRanks / (float)dupCount + 1;
                    for (int j = i - dupCount + 1; j < i + 1; j++)
                    {
                        _episodeRewardUpdate[0, _rewardsIndexKeys[j]] = (averageRank - 1) / (_batchSize - 1) -0.5f;
                    }

                    sumRanks = 0;
                    dupCount = 0;
                }
                
                _episodeRewardMean += _episodeRewards[i];

                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;
            }
            
            _episodeRewardMean /= _batchSize;
        }

        public void ReduceNoise(float noiseStd)
        {
            _networkModel.SetNoiseStd(noiseStd);
        }

        public void Dispose()
        {
            _networkModel?.Dispose();
        }
    }
}