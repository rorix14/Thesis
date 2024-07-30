using Algorithms.RL;
using DL;
using DL.NN;
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
            // var resetSate = _env.ResetEnv();
            // _envStateSize = _env.GetObservationSize * skippedFrames;
            // _nextSate = new float[_envStateSize];
            // _currentSate = new float[_envStateSize];
            // var startSateIndex = _envStateSize - _env.GetObservationSize;
            // for (int i = 0; i < _env.GetObservationSize; i++)
            // {
            //     _currentSate[startSateIndex + i] = resetSate[i];
            // }

            var updateLayers = new Layer[]
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

            var targetLayers = new Layer[]
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