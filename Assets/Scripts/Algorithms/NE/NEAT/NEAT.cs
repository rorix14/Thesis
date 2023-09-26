namespace Algorithms.NE.NEAT
{
    public class NEAT : ES
    {
        private readonly NEATModel _neatModel;

        //Cashed variables
        private new float[] _modelPredictions;
        private new readonly float[] _episodeRewardUpdate;

        public NEAT(NEATModel neatModel, int numberOfActions, int batchSize) : base(null, numberOfActions, batchSize)
        {
            _neatModel = neatModel;
            _episodeRewardUpdate = new float[batchSize];
        }

        public override int[] SamplePopulationActions(float[,] states)
        {
            _modelPredictions = _neatModel.Predict(states);

            for (int i = 0; i < _batchSize; i++)
            {
                if (_completedAgents[i])
                {
                    _sampledActions[i] = -1;
                    continue;
                }

                var individualStartIndex = i * _numberOfActions;
                var maxAction = _modelPredictions[individualStartIndex];
                var maxIndex = 0;
                for (int j = 1; j < _numberOfActions; j++)
                {
                    var actionValue = _modelPredictions[individualStartIndex + j];
                    if (maxAction > actionValue) continue;

                    maxAction = actionValue;
                    maxIndex = j;
                }

                _sampledActions[i] = maxIndex;
            }

            return _sampledActions;
        }
        
        public override void Train()
        {
            _episodeRewardMean = 0f;
            _episodeBestReward = float.MinValue;
            for (int i = 0; i < _batchSize; i++)
            {
                var episodeReward = _episodeRewards[i];
                _episodeRewardUpdate[i] = episodeReward;
                _episodeRewardMean += episodeReward;
                _episodeBestReward = episodeReward > _episodeBestReward ? episodeReward : _episodeBestReward;
                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;
            }

            _episodeRewardMean /= _batchSize;
            _finishedIndividuals = 0;

            _neatModel.Update(_episodeRewardUpdate);
        }
    }
}