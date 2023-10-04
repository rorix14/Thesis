using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestDistributionalDQN : TestDQN
    {
        [SerializeField] private int supportSize;
        [SerializeField] private float supportMinValue;
        [SerializeField] private int supportMaxValue;

        public override string GetDescription()
        {
            return "Dist DQN, max " + supportMaxValue + ", min " + supportMinValue + ", 3 layers, " +
                   neuronNumber + " neurons, " + activationFunction + ", " + batchSize + " batch size, " + gamma +
                   " gamma, " + targetNetworkCopyPeriod + " copy network, lr " + learningRate + ", decay " + decayRate +
                   ", initialization std " + weightsInitStd;
        }

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions * supportSize, ActivationFunction.Softmax,
                    Instantiate(shader), headNumber: _env.GetNumberOfActions)
            };
            var updateModel = new NetworkModel(updateLayers, new CategoricalCrossEntropy(Instantiate(shader)),
                learningRate, decayRate);

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions * supportSize, ActivationFunction.Softmax,
                    Instantiate(shader), headNumber: _env.GetNumberOfActions)
            };
            var targetModel = new NetworkModel(targetLayers, new CategoricalCrossEntropy(Instantiate(shader)),
                learningRate, decayRate);

            _DQN = new DistributionalDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                supportSize, supportMinValue, supportMaxValue, batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}