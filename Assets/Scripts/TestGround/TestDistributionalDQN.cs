using Algorithms.RL;
using NN;
using UnityEngine;
using UnityEngine.Serialization;

namespace TestGround
{
    public class TestDistributionalDQN : TestDQN
    {
        [SerializeField] private int supportSize;
        [SerializeField] private float supportMinValue;
        [SerializeField] private int supportMaxValue;

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions * supportSize, ActivationFunction.Softmax,
                    Instantiate(shader), headNumber: _env.GetNumberOfActions)
            };
            var updateModel = new NetworkModel(updateLayers, new CategoricalCrossEntropy(Instantiate(shader)));

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions * supportSize, ActivationFunction.Softmax,
                    Instantiate(shader), headNumber: _env.GetNumberOfActions)
            };
            var targetModel = new NetworkModel(targetLayers, new CategoricalCrossEntropy(Instantiate(shader)));

            _DQN = new DistributionalDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                supportSize, supportMinValue, supportMaxValue);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}