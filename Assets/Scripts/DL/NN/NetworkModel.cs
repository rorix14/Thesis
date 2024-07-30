using NN.CPU_Single;
using UnityEngine;

namespace DL.NN
{
    public class NetworkModel
    {
        public readonly Layer[] _layers;
        private readonly NetworkLoss _lossFunction;
        protected readonly float _learningRate;
        protected readonly float _decay;
        protected float _currentLearningRate;
        protected int _iteration;
        protected readonly float _beta1;
        protected readonly float _beta2;
        protected float _bata1Corrected;
        protected float _bata2Corrected;
        protected readonly int _layersCount;

        public NetworkModel(Layer[] layers, NetworkLoss lossFunction, float learningRate = 0.005f,
            float decay = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-7f)
        {
            _layers = layers;
            _lossFunction = lossFunction;
            _learningRate = learningRate;
            _currentLearningRate = learningRate;
            _decay = decay;
            _beta1 = beta1;
            _beta2 = beta2;
            _bata1Corrected = 1.0f;
            _bata2Corrected = 1.0f;
            _layersCount = layers.Length;

            foreach (var networkLayer in layers)
            {
                networkLayer.SetOptimizerVariables(beta1, beta2, epsilon);
            }
        }

        public float[,] Predict(float[,] x)
        {
            _layers[0].Forward(x);
            for (int j = 1; j < _layersCount; j++)
            {
                _layers[j].Forward(_layers[j - 1].Output);
            }

            return (float[,])_layers[_layers.Length - 1].Output;
        }

        public virtual float[,] Update(float[,] yTarget)
        {
            if (_decay > 0)
            {
                _currentLearningRate = _learningRate * (1.0f / (1.0f + _decay * _iteration));
            }

            ++_iteration;
            _bata1Corrected *= _beta1;
            _bata2Corrected *= _beta2;
            _currentLearningRate = _currentLearningRate * Mathf.Sqrt(1 - _bata2Corrected) / (1 - _bata1Corrected);

            _lossFunction.Backward((float[,])_layers[_layersCount - 1].Output, yTarget);
            _layers[_layersCount - 1]
                .Backward(_lossFunction.DInputs, _currentLearningRate);
            for (int j = _layersCount - 2; j >= 0; j--)
            {
                _layers[j].Backward(_layers[j + 1].DInput, _currentLearningRate);
            }
            // if (_iteration % 10 == 0)
            // {
            //     for (int i = 0; i < _layers.Length; i++)
            //     {
            //         var weights = new float[_layers[i]._weights.GetLength(0), _layers[i]._weights.GetLength(1)]; 
            //         _layers[i].weightsTestBuffer.GetData(weights);
            //     }
            // }

            // return (float[,])_layers[0].DInput;
            //TODO: must return the last DInput for algorithms that use Duelling architectures to work
            return null;
        }

        // Made to be used in supervised learning problems
        public void Train(int epochs, float[,] x, float[,] yTarget, int printEvery = 100)
        {
            var accuracyPrecision = NnMath.StandardDivination(yTarget) / 250;

            _iteration = 0;
            for (int i = 0; i < epochs; i++)
            {
                _layers[0].Forward(x);
                for (int j = 1; j < _layersCount; j++)
                {
                    _layers[j].Forward(_layers[j - 1].Output);
                }

                if (i % printEvery == 0)
                {
                    var networkOutput = (float[,])_layers[_layersCount - 1].Output;
                    _lossFunction.Calculate(networkOutput, yTarget);

                    float loss = 0;
                    foreach (var t in _lossFunction.SampleLosses)
                    {
                        loss += t;
                    }

                    loss /= _lossFunction.SampleLosses.Length;

                    var accuracy = 0.0f;
                    for (int j = 0; j < yTarget.GetLength(0); j++)
                    {
                        for (int k = 0; k < yTarget.GetLength(1); k++)
                        {
                            accuracy += Mathf.Abs(networkOutput[j, k] - yTarget[j, k]) < accuracyPrecision ? 1 : 0;
                        }
                    }

                    Debug.Log("(GPU) At " + i + ", loss: " + loss + ", accuracy: " + accuracy / yTarget.GetLength(0));
                }

                Update(yTarget);
            }
        }

        public float[] Loss(float[,] yTarget)
        {
            _lossFunction.Calculate((float[,])_layers[_layersCount - 1].Output, yTarget);
            return _lossFunction.SampleLosses;
        }

        public void CopyModel(NetworkModel otherModel)
        {
            //TODO: for safety reasons should check if: layers layer size is the same, weights and biases matrices match
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i].CopyLayer(otherModel._layers[i]);
            }
        }

        public void SetLossParams(float[] parameters)
        {
            var msePrio = (MeanSquaredErrorPrioritized)_lossFunction;
            msePrio?.SetLossExternalParameters(parameters);
        }

        public void Dispose()
        {
            // _lossFunction.Dispose();
            foreach (var layer in _layers)
            {
                layer.Dispose();
            }
        }
    }
}