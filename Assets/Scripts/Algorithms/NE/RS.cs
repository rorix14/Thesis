namespace Algorithms.NE
{
    public class RS : ES
    {
        private float _previousBestReward;

        public RS(ESModel networkModel, int numberOfActions, int batchSize) : base(networkModel, numberOfActions,
            batchSize)
        {
            _previousBestReward = float.MinValue;
        }

        public override void Train()
        {
            _episodeRewardMean = 0f;

            var maxReward = float.MinValue;
            var maxIndex = 0;

            for (int i = 0; i < _batchSize; i++)
            {
                var reward = _episodeRewards[i];
                _episodeRewardMean += reward;
                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;

                if (maxReward > reward) continue;

                maxReward = reward;
                maxIndex = i;
            }


            _episodeRewardMean /= _batchSize;

            _finishedIndividuals = 0;

            if (_previousBestReward >= maxReward)
            {
                maxIndex = -1;
            }

            _previousBestReward = maxReward;

            //_networkModel.SetBestIndex(maxIndex);
            _networkModel.TestUpdate(_episodeRewardUpdate, maxIndex);
        }
    }
}