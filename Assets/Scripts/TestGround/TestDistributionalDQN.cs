using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestDistributionalDQN : TestDQN
    {
        [SerializeField] private int _supportSize;
        [SerializeField] private float _supportMinValue;
        [SerializeField] private int _supportMaxValue;

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(512, _env.GetNumberOfActions * _supportSize, ActivationFunction.Linear,
                    Instantiate(shader), headNumber: _env.GetNumberOfActions)
            };
            var updateModel = new NetworkModel(updateLayers, new CategoricalCrossEntropy(Instantiate(shader)));

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(512, _env.GetNumberOfActions * _supportSize, ActivationFunction.Linear,
                    Instantiate(shader), headNumber: _env.GetNumberOfActions)
            };
            var targetModel = new NetworkModel(targetLayers, new CategoricalCrossEntropy(Instantiate(shader)));

            _DQN = new DistributionalDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                _supportSize, _supportMinValue, _supportMaxValue);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}