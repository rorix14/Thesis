using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestRainbowDQN : TestDQN
    {
        [SerializeField] private int stepNumber;
        [SerializeField] protected ComputeShader noisyShader;
        [SerializeField] private int supportSize;
        [SerializeField] private float supportMinValue;
        [SerializeField] private int supportMaxValue;

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var inputLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
            };
            var inputModel = new NetworkModel(inputLayers, new NoLoss(Instantiate(shader)));

            //Value model might not need to have a noisy shader since it does not affect the chosen action
            var valueLayers = new NetworkLayer[]
            {
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, supportSize, ActivationFunction.Linear, Instantiate(shader)),
            };
            var valueModel = new NetworkModel(valueLayers, new NoLoss(Instantiate(shader)));

            var advantageLayers = new NetworkLayer[]
            {
                new NoisyNetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(noisyShader)),
                new NoisyNetworkLayer(128, _env.GetNumberOfActions * supportSize, ActivationFunction.Linear,
                    Instantiate(noisyShader))
            };
            var advantageModel = new NetworkModel(advantageLayers, new NoLoss(Instantiate(shader)));

            DuellingNetwork updateModel = new DuellingNetwork(inputModel, valueModel, advantageModel);

            // target creation
            var inputTargetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
            };
            var inputTargetModel = new NetworkModel(inputTargetLayers, new NoLoss(Instantiate(shader)));

            var valueTargetLayers = new NetworkLayer[]
            {
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, supportSize, ActivationFunction.Linear, Instantiate(shader)),
            };
            var valueTargetModel = new NetworkModel(valueTargetLayers, new NoLoss(Instantiate(shader)));

            var advantageTargetLayers = new NetworkLayer[]
            {
                new NoisyNetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(noisyShader)),
                new NoisyNetworkLayer(128, _env.GetNumberOfActions * supportSize, ActivationFunction.Linear,
                    Instantiate(noisyShader))
            };
            var advantageTargetModel = new NetworkModel(advantageTargetLayers, new NoLoss(Instantiate(shader)));

            DuellingNetwork targetModel = new DuellingNetwork(inputTargetModel, valueTargetModel, advantageTargetModel);

            // Init rainbow model here
            _DQN = new RainbowDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                stepNumber, supportSize, supportMinValue, supportMaxValue);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}