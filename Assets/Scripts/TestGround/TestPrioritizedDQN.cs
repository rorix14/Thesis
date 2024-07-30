using Algorithms.RL;
using DL;
using DL.NN;
using UnityEngine;

namespace TestGround
{
    public class TestPrioritizedDQN : TestDQN
    {
        [SerializeField] private float beta;
        [SerializeField] private float alpha;

        public override string GetDescription()
        {
            return "PER DQN, alpha " + alpha + ", beta " + beta + ", 3 layers, " + neuronNumber + " neurons, " +
                   activationFunction + ", " + batchSize + " batch size, " + gamma + " gamma, " +
                   targetNetworkCopyPeriod + "  copy network, lr " + learningRate + ", decay " + decayRate +
                   ", initialization std " + weightsInitStd;
        }

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new Layer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredErrorPrioritized(Instantiate(shader)),
                learningRate, decayRate);

            var targetLayers = new Layer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelPrioritizedDQN(updateModel, targetModel, _env.GetNumberOfActions,
                _env.GetObservationSize, beta, alpha, batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}