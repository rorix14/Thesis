using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestNStepDQN : TestDQN
    {
        [SerializeField] private int stepNumber;

        public override string GetDescription()
        {
            return "N-step DQN, " + stepNumber + ", 3 layers, " + neuronNumber + " neurons, " + activationFunction +
                   ", " + batchSize + " batch size, " + gamma + " gamma, " + targetNetworkCopyPeriod +
                   "  copy network, lr " + learningRate + ", decay " + decayRate + ", initialization std " +
                   weightsInitStd;
        }

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true,
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true,
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelNStepDQN(stepNumber, updateModel, targetModel, _env.GetNumberOfActions,
                _env.GetObservationSize, batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}