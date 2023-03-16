using NN;
using NN.CPU_Single;
using UnityEngine;

namespace Algorithms.RL
{
    public class ModelDoubleDQN : ModelDQN
    {
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

            // TODO: test this implementation, see if the target values match
            MaxByRow(_networkModel.Predict(_nextStates));
            var targetPredictions = _targetModel.Predict(_nextStates);

            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _nextQ.Length; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done
                        ? experience.Reward
                        : experience.Reward + _gamma * targetPredictions[i, _nextQ[i].index];
            }

            _networkModel.Update(_yTarget);
        }
    }
}