using UnityEngine;

namespace NN
{
    public class NetworkModel
    {
        private readonly NetworkLayer[] _layers;
        private readonly NetworkLoss _lossFunction;
        private readonly float _learningRate;
        private readonly float _decay;
        private float _currentLearningRate;
        private int _iteration;

        public NetworkModel(NetworkLayer[] layers, NetworkLoss lossFunction, float learningRate = 0.005f,
            float decay = 1e-3f,
            float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-7f)
        {
            _layers = layers;
            _lossFunction = lossFunction;
            _learningRate = learningRate;
            _currentLearningRate = learningRate;
            _decay = decay;

            foreach (var networkLayer in layers)
            {
                networkLayer.SetOptimizerVariables(beta1, beta2, epsilon);
            }
        }

        public float[,] Predict(float[,] x)
        {
            _layers[0].Forward(x);
            for (int j = 1; j < _layers.Length; j++)
            {
                _layers[j].Forward(_layers[j - 1].Output);
            }

            return _layers[_layers.Length - 1].Output;
        }

        public void Update(float[,] yTarget)
        {
            if (_decay > 0)
                _currentLearningRate = _learningRate * (1.0f / (1.0f + _decay * _iteration));

            _lossFunction.Backward(_layers[_layers.Length - 1].Output, yTarget);
            _layers[_layers.Length - 1].Backward(_lossFunction.DInputs, _currentLearningRate, _iteration);
            for (int j = _layers.Length - 2; j >= 0; j--)
            {
                _layers[j].Backward(_layers[j + 1].DInputs, _currentLearningRate, _iteration);
            }

            ++_iteration;
        }

        public void Train(int epochs, float[,] x, float[,] yTarget, int printEvery=100)
        {
            _iteration = 0;
            for (int i = 0; i < epochs; i++)
            {
                _layers[0].Forward(x);
                for (int j = 1; j < _layers.Length; j++)
                {
                    _layers[j].Forward(_layers[j - 1].Output);
                }

                if (i % printEvery == 0)
                {
                   var loss = _lossFunction.Calculate(_layers[_layers.Length - 1].Output, yTarget);
                   Debug.Log("(GPU) At " + i +", loss: " + loss);
                }

                Update(yTarget);
            }
        }

        public void Dispose()
        {
            _lossFunction.Dispose();
            foreach (var layer in _layers)
                layer.Dispose();
        }
    }
}