using NN;
using NN.CPU_Single;
using UnityEngine;

namespace Algorithms.NE
{
    public class ESModel : NetworkModel
    {
        private readonly float _epsilon;
        public ESModel(NetworkLayer[] layers, NetworkLoss lossFunction, float learningRate = 0.005f,
            float decay = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1E-07f) : base(layers,
            lossFunction, learningRate, decay, beta1, beta2, epsilon)
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

        public override float[,] Update(float[,] yTarget)
        {
            if (_decay > 0)
            {
                _currentLearningRate = _learningRate * (1.0f / (1.0f + _decay * _iteration));
            }

            ++_iteration;
            _bata1Corrected *= _beta1;
            _bata2Corrected *= _beta2;

            //TODO: mean is already calculated in the model, so it can be just passed
            float rewardMean = NnMath.MatrixMean(yTarget);
            float rewardStd = NnMath.StandardDivination(yTarget, rewardMean);
            rewardStd = Mathf.Abs(rewardStd) < _epsilon ? _epsilon : rewardStd;
            
            for (int i = 0; i < _layers.Length; i++)
            {
                var layer = (ESNetworkLayer)_layers[i];
                layer.SetNeParameters(rewardMean, rewardStd);
                layer.Backward(yTarget, _currentLearningRate, _bata1Corrected, _bata2Corrected);
            }

            return null;
        }
    }
}