using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NN.CPU_Single
{
    public class OptimizerAdam
    {
        public float CurrentLearningRate => _currentLearningRate;

        private readonly float _learningRate;
        private float _currentLearningRate;
        private readonly float _decay;
        private int _iteration;
        private readonly float _epsilon;
        private readonly float _beta1;
        private readonly float _beta2;

        private readonly Dictionary<DenseLayer, float[,]> _layerToWeightsMomentum;
        private readonly Dictionary<DenseLayer, float[,]> _layerToWeightsCache;
        private readonly Dictionary<DenseLayer, float[,]> _layerToBiasesMomentum;
        private readonly Dictionary<DenseLayer, float[,]> _layerToBiasesCache;

        public OptimizerAdam(float learningRate = 0.001f, float decay = 0.0f, float epsilon = 1e-7f,
            float beta1 = 0.9f, float beta2 = 0.999f)
        {
            _currentLearningRate = learningRate;
            _learningRate = learningRate;
            _decay = decay;
            _iteration = 0;
            _epsilon = epsilon;
            _beta1 = beta1;
            _beta2 = beta2;

            _layerToWeightsMomentum = new Dictionary<DenseLayer, float[,]>();
            _layerToWeightsCache = new Dictionary<DenseLayer, float[,]>();
            _layerToBiasesMomentum = new Dictionary<DenseLayer, float[,]>();
            _layerToBiasesCache = new Dictionary<DenseLayer, float[,]>();
        }

        public void PreUpdateParams()
        {
            if (_decay > 0)
                _currentLearningRate = _learningRate * (1.0f / (1.0f + _decay * _iteration));
        }

        public void UpdateParams(DenseLayer layer)
        {
            CheckLayerInit(layer);
            
            for (int i = 0; i < layer.DWeights.GetLength(0); i++)
            {
                float weightMomentumCorrected;
                float weightCacheCorrected;
            
                for (int j = 0; j <  layer.DWeights.GetLength(1); j++)
                {
                    _layerToWeightsMomentum[layer][i, j] = _beta1 * _layerToWeightsMomentum[layer][i, j] +
                                                           (1 - _beta1) * layer.DWeights[i, j];
                    weightMomentumCorrected =
                        _layerToWeightsMomentum[layer][i, j] / (1 - Mathf.Pow(_beta1, _iteration + 1));
            
                    _layerToWeightsCache[layer][i, j] = _beta2 * _layerToWeightsCache[layer][i, j] +
                                                        (1 - _beta2) * Mathf.Pow(layer.DWeights[i, j], 2);
                    
                    weightCacheCorrected = _layerToWeightsCache[layer][i, j] /
                                           (1 - Mathf.Pow(_beta2, _iteration + 1));
            
                    layer.Weights[i, j] += -_currentLearningRate * weightMomentumCorrected /
                                            (Mathf.Sqrt(weightCacheCorrected) + _epsilon);
                }
            }

            for (int i = 0; i <  layer.DBiases.GetLength(0); i++)
            {
                float biasMomentumCorrected;
                float biasCacheCorrected;
                for (int j = 0; j <layer.DBiases.GetLength(1); j++)
                {
                    _layerToBiasesMomentum[layer][i, j] = _beta1 * _layerToBiasesMomentum[layer][i, j] +
                                                          (1 - _beta1) * layer.DBiases[i, j];
                    biasMomentumCorrected =
                        _layerToBiasesMomentum[layer][i, j] / (1 - Mathf.Pow(_beta1, _iteration + 1));
            
                    _layerToBiasesCache[layer][i, j] = _beta2 * _layerToBiasesCache[layer][i, j] +
                                                       (1 - _beta2) * Mathf.Pow(layer.DBiases[i, j], 2);
            
                    biasCacheCorrected = _layerToBiasesCache[layer][i, j] / (1 - Mathf.Pow(_beta2, _iteration + 1));
            
                    layer.Biases[i, j] += -_currentLearningRate * biasMomentumCorrected /
                                           (Mathf.Sqrt(biasCacheCorrected) + _epsilon);
                }
            }
            
            // float result = layer.DBiases.Cast<float>().Sum();
            // Debug.Log("(cpu) d_biases value sum: " + result);
            // result = layer.Biases.Cast<float>().Sum();
            // Debug.Log("(cpu) biases value sum: " + result);
            // float result =   layer.DWeights.Cast<float>().Sum();
            // Debug.Log("(cpu) d_weights value sum: " + result);
            // result =   layer.Weights.Cast<float>().Sum();
            // Debug.Log("(cpu) Weights value sum: " + result);
            // result =   layer.DInputs.Cast<float>().Sum();
            // Debug.Log("(cpu) d_inputs value sum: " + result);
        }

        public void PostUpdateParams()
        {
            ++_iteration;
        }

        private void CheckLayerInit(DenseLayer layer)
        {
            if (!_layerToBiasesMomentum.ContainsKey(layer))
            {
                _layerToWeightsMomentum.Add(layer, new float[layer.Weights.GetLength(0), layer.Weights.GetLength(1)]);
                _layerToWeightsCache.Add(layer, new float[layer.Weights.GetLength(0), layer.Weights.GetLength(1)]);
                _layerToBiasesMomentum.Add(layer, new float[layer.Biases.GetLength(0), layer.Biases.GetLength(1)]);
                _layerToBiasesCache.Add(layer, new float[layer.Biases.GetLength(0), layer.Biases.GetLength(1)]);
            }
        }
    }
}