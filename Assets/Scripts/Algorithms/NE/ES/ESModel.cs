using NN;
using NN.CPU_Single;
using UnityEngine;

namespace Algorithms.NE
{
    public class ESModel : NetworkModel
    {
        private float _rewardMean;
        private readonly float _epsilon;

        public float RewardMean => _rewardMean;

        public ESModel(NetworkLayer[] layers, float learningRate = 0.005f,
            float decay = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1E-07f) : base(layers,
            new NoLoss(null), learningRate, decay, beta1, beta2, epsilon)
        {
            _epsilon = epsilon;
        }

        public void SetNoiseStd(float noiseStd)
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                var layer = (ESNetworkLayer)_layers[i];
                layer.SetNoiseStd(noiseStd);
            }
        }

        public void TestUpdate(float[,] yTarget, int bestIndex)
        {
            if (bestIndex < 0)
            {
                for (int i = 0; i < _layers.Length; i++)
                {
                    ((ESNetworkLayer)_layers[i]).SetNoise();
                }
            }
            else
            {
                for (int i = 0; i < _layers.Length; i++)
                {
                    var layer = (ESNetworkLayer)_layers[i];
                    layer.SetBestIndex(bestIndex);
                    layer.Backward(yTarget, _currentLearningRate, _bata1Corrected, _bata2Corrected);
                }
            }
        }

        public override float[,] Update(float[,] yTarget)
        {
            if (_decay > 0)
            {
                _currentLearningRate = _learningRate * (1.0f / (1.0f + _decay * _iteration));
            }

            ++_iteration;
            _bata1Corrected *= _beta1;
            _bata2Corrected *= _beta2;

            _rewardMean = NnMath.MatrixMean(yTarget);
            float rewardStd = NnMath.StandardDivination(yTarget, _rewardMean);
            rewardStd = Mathf.Abs(rewardStd) < _epsilon ? _epsilon : rewardStd;

            for (int i = 0; i < _layers.Length; i++)
            {
                var layer = (ESNetworkLayer)_layers[i];
                layer.SetNeParameters(_rewardMean, rewardStd);
                layer.Backward(yTarget, _currentLearningRate, _bata1Corrected, _bata2Corrected);
                //_layers[i].Backward(yTarget, _currentLearningRate, _bata1Corrected, _bata2Corrected);
            }

            return null;
        }
    }
}