using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestNoisyDQN : TestDQN
    {
        [SerializeField] protected ComputeShader noisyShader;
        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.ReLu, Instantiate(shader), true),
                new NoisyNetworkLayer(128, 128, ActivationFunction.ReLu, Instantiate(noisyShader)),
                new NoisyNetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(noisyShader))
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)));

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.ReLu, Instantiate(shader), true),
                new NoisyNetworkLayer(128, 128, ActivationFunction.ReLu, Instantiate(noisyShader)),
                new NoisyNetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(noisyShader))
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)));

            _DQN = new NoisyDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}
