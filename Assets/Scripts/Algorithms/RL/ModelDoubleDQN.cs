using DL.NN;
using NN;
using NN.CPU_Single;

namespace Algorithms.RL
{
    public class ModelDoubleDQN : ModelDQN
    {
        private  float[,] _targetPredictions;
        
        public ModelDoubleDQN(NetworkModel networkModel, NetworkModel targetModel, int numberOfActions, int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f) : base(
            networkModel, targetModel, numberOfActions, stateSize, maxExperienceSize, minExperienceSize, batchSize,
            gamma)
        {
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            RandomBatch();

            MaxByRow(_networkModel.Predict(_nextStates));
            _targetPredictions = _targetModel.Predict(_nextStates);

            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _nextQ.Length; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done
                        ? experience.Reward
                        : experience.Reward + _gamma * _targetPredictions[i, _nextQ[i].index];
            }

            _networkModel.Update(_yTarget);
        }
    }
}