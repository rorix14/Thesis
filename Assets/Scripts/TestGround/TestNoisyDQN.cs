using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestNoisyDQN : TestDQN
    {
        [SerializeField] protected ComputeShader noisyShader;
        [SerializeField] private float sigma;

        public override string GetDescription()
        {
            return "Noisy DQN, sigma " + sigma + ", 3 layers, " + neuronNumber + " neurons, " + activationFunction +
                   ", " + batchSize + " batch size, " + gamma + " gamma, " + targetNetworkCopyPeriod +
                   "  copy network, lr " + learningRate + ", decay " + decayRate + ", initialization std 1";
        }

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NoisyNetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(noisyShader), sigma),
                new NoisyNetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear,
                    Instantiate(noisyShader), sigma)
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NoisyNetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(noisyShader), sigma),
                new NoisyNetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear,
                    Instantiate(noisyShader), sigma)
            };

            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new NoisyDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}