using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestDoubleDQN : TestDQN
    {
        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelDoubleDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}