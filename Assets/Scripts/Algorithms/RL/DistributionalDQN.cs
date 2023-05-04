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
            int stateSize, int supportSize, float vMin, float vMax, int maxExperienceSize = 10000,
            int minExperienceSize = 100, int batchSize = 32,
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

                    if (maxQ > qValue) continue;

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

            var targetPrediction = _targetModel.Predict(_nextStates);
            GetQValues(targetPrediction);
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));

            DistributionProjection(targetPrediction);

            _networkModel.Update(_yTarget);
        }

        private void GetQValues(float[,] mat)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                var maxValue = float.MinValue;
                var maxIndex = 0;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    var startIndex = i * _supportSize;
                    var qValue = 0f;
                    for (int k = 0; k < _supportSize; k++)
                    {
                        qValue += mat[0, startIndex + j] * _support[j];
                    }

                    if (maxValue > qValue) continue;

                    maxValue = qValue;
                    maxIndex = i;
                }

                _nextQ[i] = (maxIndex, maxValue);
            }
        }

        private void DistributionProjection(float[,] mat)
        {
            var distributions = new float[_batchSize, _supportSize];
            for (int i = 0; i < _batchSize; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                var startIndex = _nextQ[i].index * _supportSize;

                for (int j = 0; j < _supportSize; j++)
                {
                    var value = experience.Done ? experience.Reward : experience.Reward + _support[j] * _gamma;
                    var tz = Mathf.Clamp(value, _vMin, _vMax);
                    var b = (tz - _vMin) / _supportDelta;
                    var lower = (int)b;
                    var upper = Mathf.CeilToInt(b);

                    var distributionIndex = startIndex + j;
                    if (lower == upper)
                    {
                        distributions[i, lower] += mat[i, distributionIndex];
                    }
                    else
                    {
                        distributions[i, lower] += mat[i, distributionIndex] * (upper - b);
                        distributions[i, upper] += mat[i, distributionIndex] * (b - lower);
                    }
                }

                var actionIndex = experience.Action * _supportSize;
                for (int j = 0; j < _supportSize; j++)
                {
                    _yTarget[i, actionIndex + j] = distributions[i, j];
                }
            }
        }
    }
}