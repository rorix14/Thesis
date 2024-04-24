using DL.NN;
using NN;

namespace Algorithms.NE
{
    public class RS : ES
    {
        private readonly ESModel _esModel;

        private float _previousBestReward;
        private float _bestAdjustedFitness;

        public RS(NetworkModel networkModel, int numberOfActions, int batchSize, float noveltyRelevance = 0) : base(
            networkModel, numberOfActions, batchSize, noveltyRelevance)
        {
            _esModel = (ESModel)networkModel;

            _previousBestReward = float.MinValue;
            _bestAdjustedFitness = float.MinValue;
        }

        public override void Train()
        {
            _episodeRewardMean = 0f;
            _episodeBestReward = float.MinValue;
            _bestAdjustedFitness = float.MinValue;
            
            var maxIndex = 0;

            for (int i = 0; i < _batchSize; i++)
            {
                var reward = _episodeRewards[i];
                _episodeRewardMean += reward;
                _episodeBestReward = reward > _episodeBestReward ? reward : _episodeBestReward;
                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;

                var adjustedFitness =  _useNovelty ? _adjustedPopulationFitness[i] : reward;
                if (_bestAdjustedFitness > adjustedFitness) continue;

                _bestAdjustedFitness = adjustedFitness;
                maxIndex = i;
            }
            
            _episodeRewardMean /= _batchSize;
            _finishedIndividuals = 0;

            if (_previousBestReward >= _bestAdjustedFitness)
            {
                maxIndex = -1;
            }

            _previousBestReward = _bestAdjustedFitness;

            _esModel.TestUpdate(_episodeRewardUpdate, maxIndex);
        }
    }
}