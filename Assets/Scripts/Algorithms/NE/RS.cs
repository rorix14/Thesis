using NN;

namespace Algorithms.NE
{
    public class RS : ES
    {
        private readonly ESModel _esModel;

        private float _previousBestReward;

        public RS(NetworkModel networkModel, int numberOfActions, int batchSize) : base(networkModel, numberOfActions,
            batchSize)
        {
            _esModel = (ESModel)networkModel;

            _previousBestReward = float.MinValue;
        }

        public override void Train()
        {
            _episodeRewardMean = 0f;
            _episodeBestReward = float.MinValue;

            var maxIndex = 0;

            for (int i = 0; i < _batchSize; i++)
            {
                var reward = _episodeRewards[i];
                _episodeRewardMean += reward;
                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;

                if (_episodeBestReward > reward) continue;

                _episodeBestReward = reward;
                maxIndex = i;
            }


            _episodeRewardMean /= _batchSize;

            _finishedIndividuals = 0;

            if (_previousBestReward >= _episodeBestReward)
            {
                maxIndex = -1;
            }

            _previousBestReward = _episodeBestReward;
            
            _esModel.TestUpdate(_episodeRewardUpdate, maxIndex);
        }
    }
}