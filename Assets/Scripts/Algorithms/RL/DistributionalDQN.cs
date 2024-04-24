using DL.NN;
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

        //cached
        private float[,] _modelPredictions;
        private float[,] _targetPredictions;

        public DistributionalDQN(NetworkModel networkModel, NetworkModel targetModel, int numberOfActions,
            int stateSize, int supportSize, float vMin, float vMax, int maxExperienceSize = 10000,
            int minExperienceSize = 100, int batchSize = 32,
            float gamma = 0.9f) : base(networkModel, targetModel, numberOfActions, stateSize, maxExperienceSize,
            minExperienceSize, batchSize, gamma)
        {
            _yTarget = new float[batchSize, numberOfActions * supportSize];

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
        
        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize) return;

            RandomBatch();

            _targetPredictions = _targetModel.Predict(_nextStates);
            MaxByRow(_targetPredictions);
            _modelPredictions = _networkModel.Predict(_currentStates);
            NnMath.CopyMatrix(_yTarget, _modelPredictions);

            DistributionProjection();

            _networkModel.Update(_yTarget);
        }

        private void DistributionProjection()
        {
            for (int i = 0; i < _batchSize; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                var startIndex = _nextQ[i].index * _supportSize;

                var actionIndex = experience.Action * _supportSize;
                for (int j = 0; j < _supportSize; j++)
                {
                    _yTarget[i, actionIndex + j] = 0.0f;
                }

                for (int j = 0; j < _supportSize; j++)
                {
                    var value = experience.Done ? experience.Reward : experience.Reward + _support[j] * _gamma;
                    var tz = NnMath.Clamp(value, _vMin, _vMax);
                    var b = (tz - _vMin) / _supportDelta;
                    var lower = (int)b;
                    var upper = Mathf.CeilToInt(b);

                    var distributionIndex = startIndex + j;
                    if (lower == upper)
                    {
                        _yTarget[i, actionIndex + lower] += _targetPredictions[i, distributionIndex];
                    }
                    else
                    {
                        _yTarget[i, actionIndex + lower] +=
                            _targetPredictions[i, distributionIndex] * (upper - b);
                        _yTarget[i, actionIndex + upper] +=
                            _targetPredictions[i, distributionIndex] * (b - lower);
                    }
                }
            }
        }
        
        //TODO: this loss calculation should be temporary, this calculation should performed in the set loss class in the NN model
        public override float SampleLoss()
        {
            // Because all variables have been initialized in the train function there is no need to do it here
            // basically this gives the loss for the previous batch 
            if (_experiences.Count < _minExperienceSize) return 0.0f;

            float loss = 0;
            for (int i = 0; i < _batchSize; i++)
            {
                var actionIndex = _experiences[_batchIndexes[i]].Action * _supportSize;
                var result = 0.0f;
                for (int j = 0; j < _supportSize; j++)
                {
                    var outputValue = NnMath.Clamp(_modelPredictions[i, actionIndex + j], 1e-7f, 1f - 1e-7f);
                    result += _yTarget[i, actionIndex + j] * Mathf.Log(outputValue);
                }

                loss += result / _supportSize * -1f;
            }

            loss /= _batchSize;
            return loss;
        }

        protected override void MaxByRow(float[,] matrix, bool firstRow = false)
        {
            var sampleSize = firstRow ? 1 : _batchSize;
            for (int i = 0; i < sampleSize; i++)
            {
                var maxValue = float.MinValue;
                var maxIndex = 0;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    var startIndex = j * _supportSize;
                    var qValue = 0f;
                    for (int k = 0; k < _supportSize; k++)
                    {
                        qValue += matrix[i, startIndex + k] * _support[k];
                    }

                    if (maxValue > qValue) continue;

                    maxValue = qValue;
                    maxIndex = j;
                }

                _nextQ[i] = (maxIndex, maxValue);
            }
        }
    }
}

//TODO: must finish this to account for terminal states
// Another way to do relocate the mass of the distribution
// for (int i = 0; i < _batchSize; i++)
// {
//     var experience = _experiences[_batchIndexes[i]];
//     var startIndex = _nextQ[i].index * _supportSize;
//
//     for (int k = 0; k < _supportSize; k++)
//     {
//         distributions[i, k] = mat[i, startIndex + k];
//     }
//
//     var b = Mathf.RoundToInt((experience.Reward - _vMin) / _supportDelta);
//     b = Mathf.Clamp(b, 0, _supportSize - 1);
//
//     if (experience.Done)
//     {
//         for (int k = 0; k < _supportSize; k++)
//         {
//             distributions[i, k] = b == k ? 1f : 0f;
//         }
//     }
//     else
//     {
//         var j = 1;
//         for (int k = b; k > 0; k--)
//         {
//             distributions[i, k] += Mathf.Pow(_gamma, j) * distributions[i, k - 1];
//             j++;
//         }
//
//         j = 1;
//         for (int k = b; k < _supportSize - 1; k++)
//         {
//             distributions[i, k] += Mathf.Pow(_gamma, j) * distributions[i, k + 1];
//             j++;
//         }
//     }
//                 
//     var sum = 0f;
//     for (int k = 0; k < _supportSize; k++)
//     {
//         sum += distributions[i, k];
//     }
//
//     var actionIndex = experience.Action * _supportSize;
//     for (int k = 0; k < _supportSize; k++)
//     {
//         _yTarget[i, actionIndex + k] = distributions[i, k] / sum;
//     }
// }