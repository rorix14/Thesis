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
        private readonly float[,] _episodeRewardUpdate;
        private float _episodeRewardMean;
        private int _finishedIndividuals;

        public float EpisodeRewardMean => _episodeRewardMean;
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
        }


        public int[] SamplePopulationActions(float[,] states)
        {
            var predictions = _networkModel.Predict(states);

            for (int i = 0; i < _batchSize; i++)
            {
                if (_completedAgents[i])
                {
                    _sampledActions[i] = -1;
                    continue;
                }
                
                var maxAction = predictions[i, 0];
                var maxIndex = 0;
                for (int j = 1; j < _numberOfActions; j++)
                {
                    var actionValue = predictions[i, j];
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
            _episodeRewardMean = 0f;
            for (int i = 0; i < _batchSize; i++)
            {
                var reward = _episodeRewards[i];
                _episodeRewardUpdate[0, i] = reward;
                _episodeRewardMean = EpisodeRewardMean + reward;

                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;
            }

            _episodeRewardMean /= _batchSize;

            _finishedIndividuals = 0;

            _networkModel.Update(_episodeRewardUpdate);
        }

        public void Dispose()
        {
            _networkModel?.Dispose();
        }
    }
}