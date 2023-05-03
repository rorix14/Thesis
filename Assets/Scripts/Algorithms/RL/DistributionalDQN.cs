using NN;
using NN.CPU_Single;
using UnityEngine;

namespace Algorithms.RL
{
    public class DistributionalDQN : ModelDQN
    {
        private readonly int _supportSize;
        private readonly float[] _support;
        private readonly float _vMin;
        private readonly float _vMax;
        private readonly float _supportDelta;
        
        public DistributionalDQN(NetworkModel networkModel, NetworkModel targetModel, int numberOfActions,
            int stateSize, int supportSize, float vMin, float vMax ,int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32,
            float gamma = 0.9f) : base(networkModel, targetModel, numberOfActions, stateSize, maxExperienceSize,
            minExperienceSize, batchSize, gamma)
        {
            _supportSize = supportSize;
            _vMin = vMin;
            _vMax = vMax;
            _supportDelta = (vMax - vMin) / (supportSize - 1);
            
            _support = new float[supportSize];
            for (int i = 0; i < supportSize; i++)
            {
                _support[i] = vMin + _supportDelta * i;
            }
        }

        public override int EpsilonGreedySample(float[] state, float eps = 0.1f)
        {
            var probability = Random.value;
            if (probability > eps)
            {
                for (int i = 0; i < _stateLenght; i++)
                {
                    _predictSate[0, i] = state[i];
                }

                var maxQ = float.MinValue;
                var maxIndex = 0;
                var predicted = _networkModel.Predict(_predictSate);
                for (int i = 0; i < _numberOfActions; i++)
                {
                    var startIndex = i * _supportSize;
                    var qValue = 0f;
                    for (int j = 0; j < _supportSize; j++)
                    {
                        qValue += predicted[0, startIndex + j] * _support[j];
                    }
                    
                    if(maxQ > qValue) continue;

                    maxQ = qValue;
                    maxIndex = i;
                }
                
                return maxIndex;
            }

            return Random.Range(0, _numberOfActions);
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize) return;

            RandomBatch();
            
            MaxByRow(_targetModel.Predict(_nextStates));
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
        }


        private void DistributionProjection()
        {
            
        }
    }
}