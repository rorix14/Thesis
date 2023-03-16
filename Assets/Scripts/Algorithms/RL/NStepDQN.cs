using NN;
using NN.CPU_Single;
using UnityEngine;

namespace Algorithms.RL
{
    public class NStepDQN : ModelDQN
    {
        private readonly int _nStep;

        public NStepDQN(int nStep, NetworkModel networkModel, NetworkModel targetModel, int numberOfActions,
            int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f) : base(
            networkModel, targetModel, numberOfActions, stateSize, maxExperienceSize, minExperienceSize, batchSize,
            gamma)
        {
            _nStep = nStep <= 0 ? 1 : nStep;
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            RandomBatch();

            MaxByRow(_targetModel.Predict(_nextStates));
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));

            for (int i = 0; i < _nextQ.Length; i++)
            {
                int batchIndex = _batchIndexes[i];
                var rewardSum = 0.0f;
                for (int j = 0; j < _nStep; j++)
                {
                    var nStepExperience = _experiences[batchIndex + j];
                    rewardSum += Mathf.Pow(_gamma, j) * nStepExperience.Reward;

                    if (nStepExperience.Done) break;
                    
                    if (j == _nStep - 1)
                    {
                        rewardSum += Mathf.Pow(_gamma, _nStep) * _nextQ[i].value;
                    }
                }

                _yTarget[i, _experiences[batchIndex].Action] = rewardSum;
            }

            _networkModel.Update(_yTarget);
        }

        protected override void RandomBatch()
        {
            Debug.Log("Random Batch in n-step");
            for (int i = 0; i < _batchSize; i++)
            {
                _batchIndexes[i] = -1;
            }

            var iterations = 0;
            do
            {
                var index = Random.Range(0, _experiences.Count - _nStep);
                var hasIndex = false;

                for (int i = 0; i < iterations + 1; i++)
                {
                    if (_batchIndexes[i] != index) continue;

                    hasIndex = true;
                    break;
                }

                if (hasIndex) continue;

                _batchIndexes[iterations] = index;

                var experienceNext = _experiences[index + _nStep];
                var experienceCurrent = _experiences[index];
                for (int i = 0; i < experienceNext.CurrentState.Length; i++)
                {
                    _nextStates[iterations, i] = experienceNext.CurrentState[i];
                    _currentStates[iterations, i] = experienceCurrent.CurrentState[i];
                }

                ++iterations;
            } while (iterations < _batchSize);
        }
    }
}