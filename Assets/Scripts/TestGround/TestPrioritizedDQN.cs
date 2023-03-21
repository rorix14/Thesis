using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestPrioritizedDQN : TestDQN
    {
        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredErrorPrioritized(Instantiate(shader)));

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)));

            _DQN = new ModelPrioritizedDQN(updateModel, targetModel, _env.GetNumberOfActions,
                _env.GetObservationSize);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
        
    }
}
